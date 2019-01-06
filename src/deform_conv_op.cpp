#define EIGEN_USE_THREADS

#include <cfloat>
#include <vector>


#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
// #include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/kernels/bounds_check.h"

#include "tensorflow/core/platform/stream_executor.h"
#include "deform_conv.h"
#include "deform_conv_op_util.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

typedef std::vector<int32> TShape;
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace {
	template <typename T>
	perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory) {
		perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
		perftools::gputools::DeviceMemory<T> typed(wrapped);
		return typed;
	}

	class CublasScratchAllocator: public perftools::gputools::ScratchAllocator {
		public:
			using Stream = ::perftools::gputools::Stream;
			using DeviceMemoryBytes = ::perftools::gputools::DeviceMemory<uint8>;
			CublasScratchAllocator(OpKernelContext* context) : context_(context) {}
			int64 GetMemoryLimitInBytes(Stream* stream) override { return -1; }
			perftools::gputools::port::StatusOr<DeviceMemoryBytes> AllocateBytes(
				Stream* stream, int64 byte_size) override {
				Tensor temporary_memory;
				Status allocation_status(context_->allocate_temp(
					DT_UINT8, TensorShape({byte_size}), &temporary_memory));
				if (!allocation_status.ok()) {
					return perftools::gputools::port::StatusOr<DeviceMemoryBytes>(
						DeviceMemoryBytes::MakeFromByteSize(nullptr, 0));
				}
				// Hold the reference of the allocated tensors until the end of the
				// allocator.
				allocated_tensors_.push_back(temporary_memory);
				return perftools::gputools::port::StatusOr<DeviceMemoryBytes>(
					DeviceMemoryBytes::MakeFromByteSize(
						temporary_memory.flat<uint8>().data(),
						temporary_memory.flat<uint8>().size()));
			}

		private:
		OpKernelContext* context_;
		std::vector<Tensor> allocated_tensors_;
	};
}  // namespace

namespace functor{
	template <typename Scalar>
	struct LaunchBatchMatMul {
		static void Launch(OpKernelContext* context, const TensorShape& in_x_shape, const TensorShape& in_y_shape, const Scalar* in_x_ptr,
						   const Scalar* in_y_ptr, bool adj_x, bool adj_y, Scalar* out) {
			constexpr perftools::gputools::blas::Transpose kTranspose = is_complex<Scalar>::value
				? perftools::gputools::blas::Transpose::kConjugateTranspose
				: perftools::gputools::blas::Transpose::kTranspose;
			perftools::gputools::blas::Transpose trans[] = {perftools::gputools::blas::Transpose::kNoTranspose, kTranspose};
			const uint64 m = in_x_shape.dim_size(adj_x ? 2 : 1);
			const uint64 k = in_x_shape.dim_size(adj_x ? 1 : 2);
			const uint64 n = in_y_shape.dim_size(adj_y ? 1 : 2);
			const uint64 batch_size = in_x_shape.dim_size(0);
			auto blas_transpose_a = trans[adj_x];
			auto blas_transpose_b = trans[adj_y];

			auto* stream = context->op_device_context()->stream();
			OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

			typedef perftools::gputools::DeviceMemory<Scalar> DeviceMemoryType;
			std::vector<DeviceMemoryType> a_device_memory;
			std::vector<DeviceMemoryType> b_device_memory;
			std::vector<DeviceMemoryType> c_device_memory;
			std::vector<DeviceMemoryType*> a_ptrs;
			std::vector<DeviceMemoryType*> b_ptrs;
			std::vector<DeviceMemoryType*> c_ptrs;
			a_device_memory.reserve(batch_size);
			b_device_memory.reserve(batch_size);
			c_device_memory.reserve(batch_size);
			a_ptrs.reserve(batch_size);
			b_ptrs.reserve(batch_size);
			c_ptrs.reserve(batch_size);
			auto* a_base_ptr = in_x_ptr;
			auto* b_base_ptr = in_y_ptr;
			// auto* c_base_ptr = out->template flat<Scalar>().data();
			auto* c_base_ptr = out;
			for (int64 i = 0; i <batch_size; ++i) {
				a_device_memory.push_back(AsDeviceMemory(a_base_ptr + i * m * k));
				b_device_memory.push_back(AsDeviceMemory(b_base_ptr + i * k * n));
				c_device_memory.push_back(AsDeviceMemory(c_base_ptr + i * m * n));
				a_ptrs.push_back(&a_device_memory.back());
				b_ptrs.push_back(&b_device_memory.back());
				c_ptrs.push_back(&c_device_memory.back());
			}

			if (batch_size == 1) {
					if (n == 1 && 
						blas_transpose_b != perftools::gputools::blas::Transpose::kConjugateTranspose &&
						blas_transpose_a != perftools::gputools::blas::Transpose::kConjugateTranspose) {

						auto gemv_trans_a = blas_transpose_a == perftools::gputools::blas::Transpose::kTranspose
							? perftools::gputools::blas::Transpose::kNoTranspose
							: perftools::gputools::blas::Transpose::kTranspose;
						bool blas_launch_status =
							stream->ThenBlasGemv(
								gemv_trans_a, adj_x ? m : k, adj_x ? k : m,
								static_cast<Scalar>(1.0), *(a_ptrs[0]),
								adj_x ? m : k, *(b_ptrs[0]), 1,
								static_cast<Scalar>(0.0), c_ptrs[0], 1).ok();
						if (!blas_launch_status) {
							context->SetStatus(errors::Internal(
								"Blas xGEMV launch failed : a.shape=", in_x_shape.DebugString(),
								", b.shape=", in_y_shape.DebugString(), ", m=", m, ", n=", n,
								", k=", k));
						}
					}
					else {
						bool blas_launch_status =
							stream->ThenBlasGemm(
								blas_transpose_b, blas_transpose_a, n, m, k,
								static_cast<Scalar>(1.0), *(b_ptrs[0]),
								adj_y ? k : n, *(a_ptrs[0]), adj_x ? m : k,
								static_cast<Scalar>(0.0), c_ptrs[0], n).ok();
						if (!blas_launch_status) {
							context->SetStatus(errors::Internal(
								"Blas xGEMM launch failed : a.shape=", in_x_shape.DebugString(),
								", b.shape=", in_y_shape.DebugString(), ", m=", m, ", n=", n,
								", k=", k));
						}
					}
			}
			else {
				CublasScratchAllocator scratch_allocator(context);
				bool blas_launch_status =
					stream->ThenBlasGemmBatchedWithScratch(
						blas_transpose_b, blas_transpose_a, n, m, k,
						static_cast<Scalar>(1.0), b_ptrs, adj_y ? k : n, a_ptrs,
						adj_x ? m : k, static_cast<Scalar>(0.0), c_ptrs, n,
						batch_size, &scratch_allocator).ok();
				if (!blas_launch_status) {
					context->SetStatus(errors::Internal(
						"Blas xGEMMBatched launch failed : a.shape=",
						in_x_shape.DebugString(),
						", b.shape=", in_y_shape.DebugString(), ", m=", m, ", n=", n,
						", k=", k, ", batch_size=", batch_size));
				}
			}
		} // void Launch
	}; // struct LaunchBatchMatMul
} // namespace functor



REGISTER_OP("DeformConv2dOp")
	.Input("x: T")
	.Input("filter: T")
	.Input("offset: T")
	.Output("output: T")
	.Attr("T: {half, float, double}")
	.Attr("strides: list(int)")
	.Attr("rates: list(int)")
	.Attr("deform_group: int")
	.Attr(GetPaddingAttrString())
	.Attr("data_format: {'NHWC', 'NCHW'} = 'NCHW' ")
	.SetShapeFn([](InferenceContext * c) {
		ShapeHandle input_shape;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
		ShapeHandle filter_shape;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &filter_shape));
		ShapeHandle offset_shape;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 4, &offset_shape));

		std::vector<int32> strides;
		TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));
		if (strides.size() != 4) {
			return errors::InvalidArgument("Conv op requires stride attribute to be 4 dim, but "
										   "got: ", strides.size());
		}
		std::vector<int32> rates;
		TF_RETURN_IF_ERROR(c->GetAttr("rates", &rates));
		if (rates.size() != 4) {
			return errors::InvalidArgument("Conv op requires rate attribute to be 4 dim, but "
										   "got: ", rates.size());
		}

		string data_format;
		TensorFormat data_format_;
		TF_RETURN_IF_ERROR(c->GetAttr("data_format", &data_format));
		FormatFromString(data_format, &data_format_);
		Padding padding;
		TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

		const int32 stride_rows = GetTensorDim(strides, data_format_, 'H');
		const int32 stride_cols = GetTensorDim(strides, data_format_, 'W');

		const int32 rate_rows = GetTensorDim(rates, data_format_, 'H');
		const int32 rate_cols = GetTensorDim(rates, data_format_, 'W');

		int deform_group;
    	TF_RETURN_IF_ERROR(c->GetAttr("deform_group", &deform_group));

		DimensionHandle batch_size_dim = c->Dim(input_shape, 0);
		DimensionHandle in_depths_dim = c->Dim(input_shape, 1);    
		DimensionHandle in_rows_dim = c->Dim(input_shape, 2);
		DimensionHandle in_cols_dim = c->Dim(input_shape, 3);
		DimensionHandle output_depth_dim = c->Dim(filter_shape, 0);
		DimensionHandle filter_depth_dim = c->Dim(filter_shape, 1);
		DimensionHandle filter_rows_dim = c->Dim(filter_shape, 2);
		DimensionHandle filter_cols_dim = c->Dim(filter_shape, 3);

		auto output_channel = c->Value(output_depth_dim);
		if (output_channel % deform_group != 0) {
			return errors::InvalidArgument("Output channels cannot divided by deform_group.");
		}
	   
		

		// if (!c->ValueKnown(in_rows_dim) || !c->ValueKnown(in_rows_dim) ||
		// 	!c->ValueKnown(filter_rows_dim) || !c->ValueKnown(filter_rows_dim)) {
		// 	ShapeHandle output_shape = c->MakeShape({batch_size_dim, output_depth_dim, 
		// 											 InferenceContext::kUnknownDim,
		// 											 InferenceContext::kUnknownDim});
		// 	c->set_output(0, output_shape);
		// 	return Status::OK();
		// }

		auto in_rows = c->Value(in_rows_dim);
		auto in_cols = c->Value(in_cols_dim);
		auto filter_rows = c->Value(filter_rows_dim);
		auto filter_cols = c->Value(filter_cols_dim);
		auto eff_filter_rows = filter_rows + (stride_rows - 1) * (rate_rows - 1);
		auto eff_filter_cols = filter_cols + (stride_cols - 1) * (rate_cols - 1);
		
		int64 output_rows, output_cols;
		int64 padding_before, padding_after;
		TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerbose(
							   in_rows, eff_filter_rows, stride_rows, padding, &output_rows,
							   &padding_before, &padding_after));
		TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerbose(
							   in_cols, eff_filter_cols, stride_cols, padding, &output_cols,
							   &padding_before, &padding_after));

		ShapeHandle output_shape = c->MakeShape({batch_size_dim, output_depth_dim, output_rows, output_cols});
		c->set_output(0, output_shape);
		return Status::OK();
	});


template <typename Device, typename T>
class DeformConv2dOp : public OpKernel {
public:
	explicit DeformConv2dOp(OpKernelConstruction * context) : OpKernel(context) {
		OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
		OP_REQUIRES_OK(context, context->GetAttr("rates", &rates_));
		OP_REQUIRES_OK(context, context->GetAttr("deform_group", &deform_group_));
		string data_format;
		OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
		OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
					errors::InvalidArgument("Invalid data format"));
		CHECK(data_format_ == FORMAT_NCHW) << "Generic conv implementation only "
									  "supports NCHW tensor format for now.";
		OP_REQUIRES(context, strides_.size() == 4,
					errors::InvalidArgument("Sliding window strides field must "
											"specify 4 dimensions"));
		const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
		const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');
		const int64 rate_n = GetTensorDim(rates_, data_format_, 'N');
		const int64 rate_c = GetTensorDim(rates_, data_format_, 'C');
		OP_REQUIRES(
			context, stride_n == 1 && stride_c == 1,
			errors::InvalidArgument("Current implementation does not yet support "
									"strides in the batch and depth dimensions."));
		OP_REQUIRES(
			context, rate_n == 1 && rate_c == 1,
			errors::InvalidArgument("Current implementation does not yet support "
									"rates in the batch and depth dimensions."));
		OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
		// const int64 stride_H = GetTensorDim(strides_, data_format_, 'H');
		// const int64 stride_W = GetTensorDim(strides_, data_format_, 'W');
	}

	void Compute(OpKernelContext * context) override {
		const Tensor & input = context->input(0);
		const Tensor & filter = context->input(1);
		const Tensor & offset = context->input(2);
		const TensorShape & input_shape = input.shape();
		const TensorShape & filter_shape = filter.shape();
		const TensorShape & offset_shape = offset.shape();
		const int output_channels = filter.dim_size(0);
		OP_REQUIRES(context, input.dims() == 4,
					errors::InvalidArgument("input must be 4-dimensional",
										input.shape().DebugString()));
		OP_REQUIRES(context, filter.dims() == 4,
					errors::InvalidArgument("filter must be 4-dimensional: ",
											filter.shape().DebugString())); 
		OP_REQUIRES(context, offset.dims() == 4,
					errors::InvalidArgument("filter must be 4-dimensional: ",
											offset.shape().DebugString())); 
		
		const int batch = GetTensorDim(input, data_format_, 'N');
		const int input_channels = GetTensorDim(input, data_format_, 'C');
		const int input_rows = GetTensorDim(input, data_format_, 'H');
		const int input_cols = GetTensorDim(input, data_format_, 'W');

		const int kernel_rows = filter.dim_size(2);	
		const int kernel_cols = filter.dim_size(3);
			 
		const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
		const int stride_cols = GetTensorDim(strides_, data_format_, 'W');

		const int rate_rows = GetTensorDim(rates_, data_format_, 'H');
		const int rate_cols = GetTensorDim(rates_, data_format_, 'W');

		const int eff_kernel_rows = kernel_rows + (stride_rows - 1) * (rate_rows - 1);
		const int eff_kernel_cols = kernel_cols + (stride_cols - 1) * (rate_cols - 1);

		OP_REQUIRES(context, input_channels == filter.dim_size(1),
					errors::InvalidArgument("input and filter must have the same depth: ",
					input_channels, " vs ", filter.dim_size(1)));

		OP_REQUIRES(context, output_channels % deform_group_ == 0,
					errors::InvalidArgument("Output channels are not compatible with deform_group: ",
					output_channels, " vs ", deform_group_));

		OP_REQUIRES(context, offset.dim_size(1) == kernel_rows * kernel_cols * 2 * deform_group_,
					errors::InvalidArgument("Channels of offset is incorrect: ",
					offset.dim_size(1), " vs ", kernel_rows, "*", kernel_cols, "*2*", deform_group_));

		int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
		OP_REQUIRES_OK(context, GetWindowedOutputSize(input_rows, eff_kernel_rows, stride_rows, padding_, &out_rows, &pad_rows));
		OP_REQUIRES_OK(context, GetWindowedOutputSize(input_cols, eff_kernel_cols, stride_cols, padding_, &out_cols, &pad_cols));
		TShape pad_2d_shape({static_cast<int>(pad_rows), static_cast<int>(pad_cols)});
		TShape stride_2d_shape({stride_rows, stride_cols});
		TShape kernel_2d_shape({kernel_rows, kernel_cols});
		TShape rate_2d_shape({rate_rows, rate_cols});
		TensorShape output_shape = ShapeFromFormat(data_format_, batch, out_rows, out_cols, output_channels);
		auto param = DeformConvParam(kernel_2d_shape, 
									   stride_2d_shape, 
									   rate_2d_shape,
									   pad_2d_shape, 
									   output_channels, 
									   true);       
		this->param_ = &param;

		LayerSetUp(input_shape, output_shape, offset_shape);

		int M = output_channels_;
		int N = output_2d_flatten_dim_;
		int K = filter_3d_flatten_dim_;

		Tensor weight_3d;
		OP_REQUIRES(context, weight_3d.CopyFrom(filter, TensorShape({1, M, K})), 
					errors::InvalidArgument("Something went wrong with weight_3d."));
		const T * weight_3d_flatten_ptr = weight_3d.template flat<T>().data();
		Tensor * output_4d = nullptr;
		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_4d));
		T* output_4d_flatten_ptr = output_4d->template flat<T>().data();

		auto col_buf_3d_shape = TensorShape({1, K, N});
		Tensor col_buf_3d;
		OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, col_buf_3d_shape, &col_buf_3d));
		auto col_buf_3d_flatten_ptr = col_buf_3d.template flat<T>().data();
		
		const Device & d = context->eigen_device<Device>();
		auto col_buf_shape = TensorShape({K, out_rows, out_cols});
		auto input_flatten_ptr = input.template flat<T>().data();
		auto offset_flatten_ptr = offset.template flat<T>().data();

		for (int n = 0; n < batch_; n++) {
			functor::deform_im2col_2d<Device, T>()(d,
												  input_flatten_ptr + n * input_3d_flatten_dim_,
												  offset_flatten_ptr + n * offset_3d_flatten_dim_,
												  ToVector(input_shape),
												  ToVector(col_buf_shape),
												  (this->param_->kernel_2d_shape),
												  (this->param_->pad_2d_shape),
												  (this->param_->stride_2d_shape),
												  (this->param_->rate_2d_shape),
												  deform_group_,
												  col_buf_3d_flatten_ptr);
			
			T * output_3d_ptr = output_4d_flatten_ptr + n * output_3d_flatten_dim_;
			functor::LaunchBatchMatMul<T>::Launch(context, 
												  weight_3d.shape(), 
												  col_buf_3d.shape(), 
												  weight_3d_flatten_ptr, 
												  col_buf_3d_flatten_ptr, 
												  false, 
												  false, 
												  output_3d_ptr);
		}
				
		if (output_shape.num_elements() == 0) return;
	}

private:
	void LayerSetUp(const TensorShape & input_shape, 
					const TensorShape & output_shape,
					const TensorShape & offset_shape) {
		is_1x1_ = true;
		for (int i=0; i < param_->kernel_2d_shape.size(); i++) {
			is_1x1_ &= (param_->kernel_2d_shape[i] == 1 && param_->stride_2d_shape[i] == 1 && param_->pad_2d_shape[i] == 0);
			if (!is_1x1_) break;
		}
		batch_ = input_shape.dim_size(0);
		input_channels_ = input_shape.dim_size(1);
		output_channels_ = param_->output_channels;
		output_2d_flatten_dim_ = ProdShape(output_shape, 2);
		filter_3d_flatten_dim_ = input_channels_ * param_->kernel_2d_shape[0] * param_->kernel_2d_shape[1];
		input_3d_flatten_dim_ = ProdShape(input_shape, 1);
		offset_3d_flatten_dim_ = ProdShape(offset_shape, 1);
		output_3d_flatten_dim_ = ProdShape(output_shape, 1);
	}

	int batch_;
	int input_channels_;
	int output_channels_;
	int deform_group_;
	int output_2d_flatten_dim_;
	int filter_3d_flatten_dim_;
	int input_3d_flatten_dim_;
	int offset_3d_flatten_dim_;
	int output_3d_flatten_dim_;
	bool is_1x1_;
	std::vector<int32> strides_;
	std::vector<int32> rates_;
	Padding padding_;
	TensorFormat data_format_;
	DeformConvParam* param_;

};

#if GOOGLE_CUDA

#define REGISTER(T)                                                 	\
	REGISTER_KERNEL_BUILDER(                                          	\
		Name("DeformConv2dOp").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
		DeformConv2dOp<GPUDevice, T>);    

TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
// TF_CALL_half(REGISTER);

#undef REGISTER

#endif

}// namespace tensorflow





























