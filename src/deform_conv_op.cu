#ifndef TENSORFLOW_KERNELS_CONV_OPS_im2col_gpu_H_
#define TENSORFLOW_KERNELS_CONV_OPS_im2col_gpu_H_

// #if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "deform_conv.h"
#include "cuda.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

#include <algorithm>
#include <cstring>
#include <vector>
#include <stdio.h>


namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;
typedef std::vector<int32> TShape;

template <typename DType>
__device__ DType deform_im2col_bilinear(const DType* thread_input_ptr, 
										const int input_width,
        								const int thread_input_h_left,	// How many grids left in height
        								const int thread_input_w_left, 	// How many grids left in width
        								DType kernel_height_loc, 	// height w.r.t kernel start height
        								DType kernel_width_loc) {	// width w.r.t kernel start width

    int height_low = floor(kernel_height_loc);
    int width_low = floor(kernel_width_loc);
    int height_high;
    int width_high;
    if (height_low >= thread_input_h_left - 1) {
        height_high = height_low = thread_input_h_left - 1;
        kernel_height_loc = (DType)height_low;
    }
    else {
        height_high = height_low + 1;
    }

    if (width_low >= thread_input_w_left - 1) {
        width_high = width_low = thread_input_w_left - 1;
        kernel_width_loc = (DType)width_low;
    }
    else {
        width_high = width_low + 1;
    }

    DType height_low_dist = kernel_height_loc - height_low;
    DType width_low_dist = kernel_width_loc - width_low;
    DType height_high_dist = 1 - height_low_dist;
    DType width_high_dist = 1 - width_low_dist;
    //	---------
    //	| 1 | 2 |
    //	---------
    //	| 3 | 4 |
    //	---------
    DType v1 = thread_input_ptr[height_low * input_width + width_low];
    DType v2 = thread_input_ptr[height_low * input_width + width_high];
    DType v3 = thread_input_ptr[height_high * input_width + width_low];
    DType v4 = thread_input_ptr[height_high * input_width + width_high];

    DType w1 = height_high_dist * width_high_dist;
    DType w2 = height_high_dist * width_low_dist;
    DType w3 = height_low_dist * width_high_dist;
    DType w4 = height_low_dist * width_low_dist;

    DType val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
    return val;
}

// template <typename DType>
// __device__ DType deform_im2col_bilinear(const DType* bottom_data, const int data_width,
//         const int height, const int width, DType h, DType w) {

//     int h_low = floor(h);
//     int w_low = floor(w);
//     int h_high;
//     int w_high;
//     if (h_low >= height - 1) {
//         h_high = h_low = height - 1;
//         h = (DType)h_low;
//     }
//     else {
//         h_high = h_low + 1;
//     }

//     if (w_low >= width - 1) {
//         w_high = w_low = width - 1;
//         w = (DType)w_low;
//     }
//     else {
//         w_high = w_low + 1;
//     }

//     DType lh = h - h_low;
//     DType lw = w - w_low;
//     DType hh = 1 - lh, hw = 1 - lw;

//     DType v1 = bottom_data[h_low * data_width + w_low];
//     DType v2 = bottom_data[h_low * data_width + w_high];
//     DType v3 = bottom_data[h_high * data_width + w_low];
//     DType v4 = bottom_data[h_high * data_width + w_high];
//     DType w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

//     DType val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
//     return val;
// }


template <typename DType>
__global__ void deform_im2col_2d_gpu_kernel(
				const int num_kernels_per_filter, 
				const DType* batch_input_ptr, 
				const DType* batch_offset_ptr,
				const int input_height, 
				const int input_width, 
				const int kernel_height, 
				const int kernel_width,
				const int pad_height, 
				const int pad_width,
				const int stride_height, 
				const int stride_width,
				const int dilation_height,
				const int dilation_width,
				const int channel_per_deform_group,
				const int output_height, 
				const int output_width,
				DType* col_buf_3d_flatten_ptr) {

	CUDA_1D_KERNEL_LOOP(index, num_kernels_per_filter) {
		// index index of output matrix
		const int thread_output_w = index % output_width;
		const int thread_output_h = (index / output_width) % output_height;
		const int thread_channel = (index / output_width) / output_height;
		const int thread_filter_init_loc = thread_channel * kernel_height * kernel_width;

		// compute deformable group index
		const int deform_group_idx = thread_channel / channel_per_deform_group;
		const int thread_input_h = thread_output_h * stride_height - pad_height;
		const int thread_input_w = thread_output_w * stride_width - pad_width;
		// printf("%d, %d\n", pad_height, pad_width);
		const int thread_input_h_left = input_height - thread_input_h;
		const int thread_input_w_left = input_width - thread_input_w;

		DType* current_data_col_ptr = col_buf_3d_flatten_ptr + 
									  thread_filter_init_loc * output_height * output_width + 
									  thread_output_h * output_width + 
									  thread_output_w;
		const DType* thread_input_ptr = batch_input_ptr + 
										thread_channel * input_height * input_width + 
										thread_input_h * input_width + 
										thread_input_w;		
		const DType* thread_offset_ptr = batch_offset_ptr + 
										 2 * deform_group_idx * 
										 kernel_height * kernel_width *
										 output_height * output_width;
		// offset -> [2 * deform_group * kernel_width * kernel_height, output_height, output_width]

		for (int i = 0; i < kernel_height; i++) {
			for (int j = 0; j < kernel_width; j++) {
				const int offset_h_ptr = 2 * (i * kernel_width + j) * output_height * output_width + 
										 thread_output_h * output_width + 
										 thread_output_w;
				const int offset_w_ptr = 2 * (i * kernel_width + j) * output_height * output_width + 
										 output_height * output_width + 
										 thread_output_h * output_width + 
										 thread_output_w;
				const DType offset_h = thread_offset_ptr[offset_h_ptr];
				const DType offset_w = thread_offset_ptr[offset_w_ptr];
				// The datatype of following variables need to be changed into <Dtype> if the deformable conv is activated.
				const DType current_input_h = thread_input_h + i * dilation_height + offset_h;
				const DType current_input_w = thread_input_w + j * dilation_width + offset_w;
				DType val = static_cast<DType>(0);
				if (current_input_h >= 0 && current_input_w >= 0 && 
						current_input_h < input_height && current_input_w < input_width) {
					const DType kernel_height_loc = i * dilation_height + offset_h;
					const DType kernel_width_loc = j * dilation_width + offset_w;
					val = deform_im2col_bilinear(thread_input_ptr, 
												 input_width,
												 thread_input_h_left,
												 thread_input_w_left,
												 kernel_height_loc,
												 kernel_width_loc);
				}
				*current_data_col_ptr = val;
				current_data_col_ptr += output_height * output_width;
			}
		}
	}
}

namespace functor {

	inline int ProdShape(const TShape & shape, int start);
	template <typename DType>
	struct deform_im2col_2d<GPUDevice, DType> {
		void operator()(const GPUDevice& d, 				// 0 -> device
						const DType* batch_input_ptr, 		// 1 -> input data start pointer, ranging according to n
						const DType* batch_offset_ptr,
						const TShape& input_shape, 			// 2 -> input shape = [N, C, H, W]
						const TShape& col_buf_shape, 		// 3 -> shape = [filter_3d_flatten, output_rows, output_cols]
						const TShape& kernel_2d_shape,		// 4 -> kernel 2D shape
						const TShape& pad_2d_shape, 		// 5 -> padding 2D shape
						const TShape& stride_2d_shape, 		// 6 -> stride 2D shape
						const TShape& dilation_2d_shape,
						const int deform_group,
						DType* col_buf_3d_flatten_ptr) {	// 7 -> flatten col_buf_3d, shape = [1 * filter_3d_flatten_dim_ * output_2d_flatten_dim_]
			// num_axes should be smaller than block size
			int num_spatial_axes = kernel_2d_shape.size();
			int channel_per_deform_group = input_shape[1] / deform_group;
			int num_kernels_per_filter = input_shape[1] * ProdShape(col_buf_shape, 1);
			CudaLaunchConfig config = GetCudaLaunchConfig(num_kernels_per_filter, d);
			CHECK_LT(num_spatial_axes, config.thread_per_block);
			switch (num_spatial_axes) {
				case 2:
					deform_im2col_2d_gpu_kernel<DType> 
					<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
						num_kernels_per_filter, 
						batch_input_ptr, 
						batch_offset_ptr,
						input_shape[2], 
						input_shape[3], 
						kernel_2d_shape[0], 
						kernel_2d_shape[1],
						pad_2d_shape[0], 
						pad_2d_shape[1], 
						stride_2d_shape[0], 
						stride_2d_shape[1],
						dilation_2d_shape[0],
						dilation_2d_shape[1],
						channel_per_deform_group,  
						col_buf_shape[1], 
						col_buf_shape[2], 
						col_buf_3d_flatten_ptr);
					break;
				default:
					LOG(FATAL) << "im2col_nd_gpu does not support computation with "
							   << num_spatial_axes << " spatial axes";
			}
		}
	};

	inline int ProdShape(const TShape &shape, int start) {
		int64 res = 1;
		for(int i=start; i<shape.size(); i++) {
			res*=shape[i];
		}
		return res;
	}
}


#define DECLARE_GPU_SPEC(DType)                                  \
	template struct functor::deform_im2col_2d<GPUDevice, DType>; 
	
// extern template struct Copy<GPUDevice, T>;
TF_CALL_float(DECLARE_GPU_SPEC);
TF_CALL_double(DECLARE_GPU_SPEC);
// TF_CALL_half(DECLARE_GPU_SPEC);

// TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
#undef DECLARE_GPU_SPEC


}

#endif  // TENSORFLOW_KERNELS_CONV_OPS_im2col_gpu_H_
