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




REGISTER_OP("DeformConv2dGradOp")
	.Input("x: T")
	.Input("filter: T")
	.Input("offset: T")
	.Input("out_grad: T")
	.Output("x_grad: T")
	.Output("filter_grad: T")
	.Output("offset_grad: T")
	.Attr("T: {half, float, double}")
	.Attr("strides: list(int)")
	.Attr("rates: list(int)")
	.Attr("deform_group: int")
	.Attr(GetPaddingAttrString())
	.Attr("data_format: {'NHWC', 'NCHW'} = 'NCHW' ")
	.SetShapeFn([](InferenceContext * c) {
		c->set_output(0, c->input(0));
		c->set_output(1, c->input(1));
		c->set_output(2, c->input(2));
		return Status::OK();
	});

template <typename Device, typename T>
class DeformConv2dGradOp : public OpKernel {
public:
	explicit DeformConv2dGradOp(OpKernelConstruction * context) : OpKernel(context) {
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
        OP_REQUIRES(context, rates_.size() == 4,
                    errors::InvalidArgument("Sliding window rates field must "
                                            "specify 4 dimensions"));
        const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
        const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');
        OP_REQUIRES(
            context, stride_n == 1 && stride_c == 1,
            errors::InvalidArgument("Current implementation does not yet support "
                                    "strides in the batch and depth dimensions."));
        const int64 rate_n = GetTensorDim(rates_, data_format_, 'N');
        const int64 rate_c = GetTensorDim(rates_, data_format_, 'C');
        OP_REQUIRES(
            context, rate_n == 1 && rate_c == 1,
            errors::InvalidArgument("Current implementation does not yet support "
                                    "rates in the batch and depth dimensions."));
        OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
	} // explicit ConvOpGrad

	void Compute(OpKernelContext * context) override {
		const Tensor & input = context->input(0);
		const Tensor & filter = context->input(1);
		const Tensor & offset = context->input(2);
		const Tensor & output_grad = context->input(3);
		const T* input_ptr = input.template flat<T>().data();
	    const T* filter_ptr = filter.template flat<T>().data();
	    const T* offset_ptr = offset.template flat<T>().data();
	    const T* output_grad_ptr = output_grad.template flat<T>().data();
	    const TensorShape& input_shape = input.shape();
	    const TensorShape& filter_shape = filter.shape();
	    const TensorShape& offset_shape = offset.shape();
	    const TensorShape& output_grad_shape = output_grad.shape();

	    const int output_channels = filter.dim_size(0);

	    const int batch = GetTensorDim(input, data_format_, 'N');
		const int input_channels = GetTensorDim(input, data_format_, 'C');
		const int input_rows = GetTensorDim(input, data_format_, 'H');
		const int input_cols = GetTensorDim(input, data_format_, 'W');

		const int output_rows = GetTensorDim(output_grad, data_format_, 'H');
		const int output_cols = GetTensorDim(output_grad, data_format_, 'W');

	    const int kernel_rows = filter.dim_size(2);	
	    const int kernel_cols = filter.dim_size(3);
	         
	    const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
    	const int stride_cols = GetTensorDim(strides_, data_format_, 'W');

    	const int rate_rows = GetTensorDim(rates_, data_format_, 'H');
    	const int rate_cols = GetTensorDim(rates_, data_format_, 'W');

    	const int eff_kernel_rows = kernel_rows + (stride_rows - 1) * (rate_rows - 1);
    	const int eff_kernel_cols = kernel_cols + (stride_cols - 1) * (rate_cols - 1);

    	OP_REQUIRES(context, output_channels % deform_group_ == 0,
    		errors::InvalidArgument("Output channels can not be divided by deform_group."));

    	int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
	    OP_REQUIRES_OK(context, GetWindowedOutputSize(input_rows, eff_kernel_rows, stride_rows, padding_, &out_rows, &pad_rows));
	    OP_REQUIRES_OK(context, GetWindowedOutputSize(input_cols, eff_kernel_cols, stride_cols, padding_, &out_cols, &pad_cols));
	    OP_REQUIRES(context, out_rows == output_rows && out_cols == output_cols,
	    	errors::InvalidArgument("The dimension of input 'output_grad' does not match the expected dimension, "
	    							"Input output_grad: ", output_rows, " vs expected output: ", out_rows));
	    TShape pad_2d_shape({static_cast<int>(pad_rows), static_cast<int>(pad_cols)});
	    TShape stride_2d_shape({stride_rows, stride_cols});
	    TShape kernel_2d_shape({kernel_rows, kernel_cols});
	    TShape rate_2d_shape({rate_rows, rate_cols});
	    TensorShape output_shape = ShapeFromFormat(data_format_, batch, out_rows, out_cols, output_channels);
	    auto param = DeformConvParam(kernel_2d_shape, stride_2d_shape, rate_2d_shape, pad_2d_shape, output_channels, true);       
	    param_ = &param;

	    LayerSetUp(input_shape, output_shape, offset_shape);

	    int M = filter_3d_flatten_dim_;
	    int N = output_2d_flatten_dim_;
	    int K = output_channels_;

	    Tensor * input_grad = nullptr;
	    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &input_grad));
	    auto input_grad_ptr = input_grad->template flat<T>().data();

	    Tensor * filter_grad = nullptr;
	    OP_REQUIRES_OK(context, context->allocate_output(1, filter.shape(), &filter_grad));
	    auto filter_grad_ptr = filter_grad->template flat<T>().data();

	    Tensor * offset_grad = nullptr;
	    OP_REQUIRES_OK(context, context->allocate_output(2, offset.shape(), &offset_grad));
	    auto offset_grad_ptr = offset_grad->template flat<T>().data();

	    TensorShape col_buf_shape({input_channels * kernel_rows * kernel_cols, output_grad.dim_size(2), output_grad.dim_size(3)});
	    TensorShape col_buf_3d_shape({1, M, N});
	    Tensor col_buf_3d;
	    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, col_buf_3d_shape, &col_buf_3d));
	    auto col_buf_3d_flatten_ptr = col_buf_3d.template flat<T>().data();

	    Tensor temp_filter_grad;
	    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, filter.shape(), &temp_filter_grad));
    	auto temp_filter_grad_ptr = temp_filter_grad.template flat<T>().data();    

	    auto weight_3d_shape = TensorShape({1, K, M});
	    const T* weight_3d_flatten_ptr = filter_ptr;
	    auto output_grad_3d_shape = TensorShape({1, K, N});
	    const Device& d = context->eigen_device<Device>();

	    functor::setZero<Device, T>()(d, 1 * M * N, col_buf_3d_flatten_ptr);
	    functor::setZero<Device, T>()(d, ProdShape(input.shape(), 0), input_grad_ptr);
	    functor::setZero<Device, T>()(d, ProdShape(filter.shape(), 0), filter_grad_ptr);
	    functor::setZero<Device, T>()(d, ProdShape(filter.shape(), 0), temp_filter_grad_ptr);
	    functor::setZero<Device, T>()(d, ProdShape(offset.shape(), 0), offset_grad_ptr);

	    for (int n = 0; n < batch_; n++) {
	    	functor::LaunchBatchMatMul<T>::Launch(context, 
	    										  weight_3d_shape,
	    										  output_grad_3d_shape,
	    										  weight_3d_flatten_ptr,
	    										  output_grad_ptr + n * output_grad_3d_flatten_dim_,
	    										  true,
	    										  false,
	    										  col_buf_3d_flatten_ptr);
	    	functor::deform_col2im_2d<Device, T>()(d,
				    							   col_buf_3d_flatten_ptr,
				    							   offset_ptr + n * offset_3d_flatten_dim_,
				    							   ToVector(input_shape),
				    							   ToVector(col_buf_shape),
				    							   param_->kernel_2d_shape,
				    							   param_->pad_2d_shape,
				    							   param_->stride_2d_shape,
				    							   param_->rate_2d_shape,
				    							   deform_group_,
				    							   input_grad_ptr + n * input_3d_flatten_dim_);
	    	functor::deform_col2im_offset_2d<Device, T>()(d,
	    												col_buf_3d_flatten_ptr,
	    												input_ptr + n * input_3d_flatten_dim_,
	    												offset_ptr + n * offset_3d_flatten_dim_,
	    												ToVector(input_shape),
	    												ToVector(col_buf_shape),
	    												param_->kernel_2d_shape,
					    							    param_->pad_2d_shape,
					    							    param_->stride_2d_shape,
					    							    param_->rate_2d_shape,
					    							    deform_group_,
					    							    offset_grad_ptr + n * offset_3d_flatten_dim_);
	    	functor::deform_im2col_2d<Device, T>()(d,
	    										   input_ptr + n * input_3d_flatten_dim_,
	    										   offset_ptr + n * offset_3d_flatten_dim_,
	    										   ToVector(input_shape),
	    										   ToVector(col_buf_shape),
	    										   param_->kernel_2d_shape,
	    										   param_->pad_2d_shape,
	    										   param_->stride_2d_shape,
	    										   param_->rate_2d_shape,
	    										   deform_group_,
	    										   col_buf_3d_flatten_ptr);
	    	if (n == 0) {
	    		functor::LaunchBatchMatMul<T>::Launch(context,
	    											  output_grad_3d_shape,
	    											  col_buf_3d_shape,
	    											  output_grad_ptr + n * output_grad_3d_flatten_dim_,
	    											  col_buf_3d_flatten_ptr,
	    											  false,
	    											  true,
	    											  filter_grad_ptr);
	    	}
	    	else {
	    		functor::LaunchBatchMatMul<T>::Launch(context,
	    											  output_grad_3d_shape,
	    											  col_buf_3d_shape,
	    											  output_grad_ptr + n * output_grad_3d_flatten_dim_,
	    											  col_buf_3d_flatten_ptr,
	    											  false,
	    											  true,
	    											  temp_filter_grad_ptr);
	    		functor::pureAddTo<Device, T>()(d,
	    										ProdShape(filter_shape, 0),
	    										filter_grad_ptr,
	    										temp_filter_grad_ptr);
	    	}
	    }

	} //void Compute

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
		output_grad_3d_flatten_dim_ = ProdShape(output_shape, 1);


	}
	int batch_;
	int input_channels_;
	int output_channels_;
	int filter_3d_flatten_dim_;
	int input_3d_flatten_dim_;
	int offset_3d_flatten_dim_;
	int output_grad_3d_flatten_dim_;
	int output_2d_flatten_dim_;
	int deform_group_;
	bool is_1x1_;
	bool no_bias;
	std::vector<int32> strides_;
	std::vector<int32> rates_;
    Padding padding_;
    TensorFormat data_format_;
    DeformConvParam* param_;



}; //class ConvOpGrad

#if GOOGLE_CUDA

#define REGISTER(T)                                                 	\
	REGISTER_KERNEL_BUILDER(                                          	\
		Name("DeformConv2dGradOp").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
		DeformConv2dGradOp<GPUDevice, T>);    

TF_CALL_float(REGISTER);
TF_CALL_double(REGISTER);
// TF_CALL_half(REGISTER);

#undef REGISTER

#endif

}// namespace tensorflow





















