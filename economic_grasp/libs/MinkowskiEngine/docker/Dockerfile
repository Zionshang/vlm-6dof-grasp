# If use previous versions, modify these variables
# ARG PYTORCH="1.9.0"
# ARG CUDA="11.1"
# ARG CUDNN="8"

ARG PYTORCH="2.7.0"
ARG CUDA="12.6"
ARG CUDNN="9"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

ARG CUDA_SM_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.9 9.0 12.0"


# Install dependencies
RUN apt-get update
RUN apt-get install -y git ninja-build cmake build-essential libopenblas-dev \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core

# Build
WORKDIR /workspace
RUN git clone https://github.com/CiSong10/MinkowskiEngine.git && \
    cd MinkowskiEngine && \
    git checkout cuda12-installation 

WORKDIR ./MinkowskiEngine

# Apply the fix for error: Compilation: compiler finds both std::to_address and cuda::std::to_address (https://github.com/NVIDIA/MinkowskiEngine/issues/596)
RUN sed -i 's/\bauto __raw = __to_address(__r.get());/auto __raw = std::__to_address(__r.get());/' /usr/include/c++/11/bits/shared_ptr_base.h
RUN python setup.py install --force_cuda --blas=openblas
