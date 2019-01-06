#include "tensorflow/core/util/tensor_format.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
typedef std::vector<int32> TShape;

inline int ProdShape(const TensorShape &shape, int start) {
    int64 res = 1;
    for(int i=start; i<shape.dims(); i++) {
        res*=shape.dim_size(i);
    }
    return res;
}

inline std::vector<int> ToVector(const TensorShape &shape) {
    // int64 res = 1;
    std::vector<int> res;
    for(int i=0; i<shape.dims(); i++) {
        res.push_back(shape.dim_size(i));
    }
    return res;
}

inline TShape ToVector(const TShape &shape) {
    // int64 res = 1;
    return shape;
}

inline TensorShape Slice(const TensorShape &shape, int start, int end) {
    TensorShape temp = shape;
    for(int i=0; i<start; i++) {
        temp.RemoveDim(0);
    }
    for(int i=0; i<shape.dims()-end; i++) {
        temp.RemoveDim(temp.dims()-1);
    }
    return temp;
}

struct DeformConvParam {

    DeformConvParam(TShape kernel_2d_shape, 
              TShape stride_2d_shape,
              TShape rate_2d_shape,
              TShape pad_2d_shape, 
              int output_channels,
              bool no_bias): 
                kernel_2d_shape(kernel_2d_shape), 
                stride_2d_shape(stride_2d_shape), 
                rate_2d_shape(rate_2d_shape),
                pad_2d_shape(pad_2d_shape), 
                output_channels(output_channels), 
                no_bias(no_bias) {};

    TShape kernel_2d_shape, stride_2d_shape, rate_2d_shape, pad_2d_shape;
    int output_channels;
    bool no_bias;
};

}