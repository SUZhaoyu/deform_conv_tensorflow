TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
echo $TF_INC
if [ ! -f $TF_INC/tensorflow/stream_executor/cuda/cuda_config.h ]; then
    cp ./cuda_config.h $TF_INC/tensorflow/stream_executor/cuda/
fi
CUDA_HOME=/usr/local/cuda
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

nvcc -ccbin=/usr/bin/g++-4.9 -std=c++11 -c -o build/deform_conv_op.cu.o src/deform_conv_op.cu -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -L /usr/local/cuda-9.0/lib64/ -I /usr/local/ -I $TF_INC/external/nsync/public --expt-relaxed-constexpr -DNDEBUG -gencode arch=compute_61,code=sm_61 
g++-4.9 -std=c++11 -shared -o build/deform_conv_op.so src/deform_conv_op.cpp build/deform_conv_op.cu.o -D_GLIBCXX_USE_CXX11_ABI=0 -I $TF_INC -fPIC -L $CUDA_HOME/lib64 -lcudart -D GOOGLE_CUDA=1 -Wfatal-errors -I $CUDA_HOME/include -L $TF_LIB -ltensorflow_framework

nvcc -ccbin=/usr/bin/g++-4.9 -std=c++11 -c -o build/deform_conv_grad_op.cu.o src/deform_conv_grad_op.cu -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -L /usr/local/cuda-9.0/lib64/ -I /usr/local/ -I $TF_INC/external/nsync/public --expt-relaxed-constexpr -DNDEBUG -gencode arch=compute_61,code=sm_61 
g++-4.9 -std=c++11 -shared -o build/deform_conv_grad_op.so src/deform_conv_grad_op.cpp build/deform_conv_grad_op.cu.o -D_GLIBCXX_USE_CXX11_ABI=0 -I $TF_INC -fPIC -L $CUDA_HOME/lib64 -lcudart -D GOOGLE_CUDA=1 -Wfatal-errors -I $CUDA_HOME/include -L $TF_LIB -ltensorflow_framework