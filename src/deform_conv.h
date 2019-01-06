/*!
 ******************* BEGIN Caffe Copyright Notice and Disclaimer ****************
 *
 * COPYRIGHT
 *
 * All contributions by the University of California:
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014-2017, the respective contributors
 * All rights reserved.
 *
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 *
 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * CONTRIBUTION AGREEMENT
 *
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 ***************** END Caffe Copyright Notice and Disclaimer ********************
 *
 * Copyright (c) 2017 by Contributors
 * \file deformable_im2col.h
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, and dilation.
 * These functions are mainly used in convolution operators.
 * The implementation of the im2col and col2im algorithms
 * are copied from Caffe with minor interface modifications
 * adapting to MXNet data structures.
 */

#ifndef TENSORFLOW_KERNELS_CONV_OPS_im2col_H_
#define TENSORFLOW_KERNELS_CONV_OPS_im2col_H_

// #define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include <cstring>
#include <vector>

namespace tensorflow {
// typedef Eigen::ThreadPoolDevice CPUDevice;
typedef std::vector<int> TShape;
// typedef Eigen::GpuDevice GPUDevice;

namespace functor {



template <typename Device, typename DType>
struct deform_im2col_2d { 
    void operator()(const Device& d,                 // 0 -> device
                    const DType* batch_input_ptr,       // 1 -> input data start pointer, ranging according to n
                    const DType* batch_offset_ptr,
                    const TShape& input_shape,          // 2 -> input shape = [N, C, H, W]
                    const TShape& col_buf_shape,        // 3 -> shape = [filter_3d_flatten, output_rows, output_cols]
                    const TShape& kernel_2d_shape,      // 4 -> kernel 2D shape
                    const TShape& pad_2d_shape,         // 5 -> padding 2D shape
                    const TShape& stride_2d_shape,      // 6 -> stride 2D shape
                    const TShape& dilation_2d_shape,
                    const int deform_group,
                    DType* col_buf_3d_flatten_ptr);
};

template <typename Device, typename DType>
struct deform_col2im_2d {
     void operator()(const Device& d,
                    const DType* col_buf_ptr,
                    const DType* offset_ptr,
                    const TShape& input_shape,
                    const TShape& col_buf_shape,    //--> vector<int>[input_channel*filter_row*filter_col, output_row, output_col]
                    const TShape& kernel_2d_shape,
                    const TShape& pad_2d_shape,
                    const TShape& stride_2d_shape,
                    const TShape& dilation_2d_shape,
                    const int deform_group,
                    DType* input_grad_ptr);
};

template <typename Device, typename DType>
struct deform_col2im_offset_2d {
  void operator()(const Device& d,
                  const DType* col_buf_ptr,
                  const DType* input_ptr,
                  const DType* offset_ptr,
                  const TShape& input_shape,
                  const TShape& col_buf_shape,
                  const TShape& kernel_2d_shape,
                  const TShape& pad_2d_shape,
                  const TShape& stride_2d_shape,
                  const TShape& dilation_2d_shape,
                  const int deform_group,
                  DType* offset_grad_ptr);
};

template <typename Device, typename DType>
struct setZero {
     void operator() (const Device& d, 
                      const int n, 
                      DType* result_data);
};

template <typename Device, typename DType>
struct pureAddTo {
     void operator() (const Device& d, 
                      const int n, 
                      DType* result_data, 
                      const DType* right_data);
};


}  // namespace functor
}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_CONV_OPS_im2col_H_
