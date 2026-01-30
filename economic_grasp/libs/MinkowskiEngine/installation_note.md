# MinkowskiEngine Installation Notes (CUDA 11.1 & CUDA 12.2)

This is a note for myself regarding the installation of MinkowskiEngine.

MinkowskiEngine can be installed with CUDA 12, but ensure that your PyTorch and CUDA versions are compatible:
* [Previous versions of PyTorch](https://pytorch.org/get-started/previous-versions/)
* [Latest Version](https://pytorch.org/get-started/locally/)
* PyTorch does not have version for CUDA 12.2, install PyTorch compatible with CUDA 12.1
* I am using PyTorch 2.4.0 because I need to install `pytorch-scatter` `pytorch-sparse` `pyg` that are built on torch 2.4.0. You can use other versions such as PyTorch 2.5.1

## Installation with Docker

Checkout the updated `Dockerfile` in my [branch](https://github.com/CiSong10/MinkowskiEngine/tree/cuda12-installation). Thanks to this [comment by @QuteSaltyFish](https://github.com/NVIDIA/MinkowskiEngine/issues/620#issuecomment-3128267425)

## Steps for Local Installation

```
mamba create -n <env_name> python=3.9
mamba activate <env_name>
mamba install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
mamba install -y openblas-devel -c anaconda
mamba install -y nvidia/label/cuda-12.1.0::cuda-toolkit
```

### Check `pip` and `setuptools` versions

I believe the `pip` and `setuptools` versions should be low enough. Here's mine:
* `pip==22.2.2`
* `setuptools==69.5.1`

### Verify CUDA is available

* **Check CUDA version**: `nvcc -V`
* **Check CuDNN version**: `cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2`

```
python -c "import torch; print(torch.cuda.is_available())"
```

### Verify GCC version

```
gcc --version
g++ --version
```

* Use GCC <= 12 for building with CUDA 12
* Use GCC <= 10 for building with CUDA 11

Be careful when running `sudo apt update` since it will make you install GCC 13. You will need to temporary export the environment for building with a different GCC version

```
export CC=gcc-12
export CXX=g++-12
```

### Ensure dependencies are met

```
sudo apt install -y build-essential cmake libopenblas-dev
```

### Compling

Use my [branch](https://github.com/CiSong10/MinkowskiEngine/tree/cuda12-installation) that allows compling based on cuda 12, according to https://github.com/NVIDIA/MinkowskiEngine/issues/601

```
git clone https://github.com/CiSong10/MinkowskiEngine.git
cd MinkowskiEngine/
git checkout cuda12-installation
```

```
python setup.py install --blas=openblas 
```

### Verify environment and MinkowskiENgine installation

```
wget -q https://raw.githubusercontent.com/NVIDIA/MinkowskiEngine/master/MinkowskiEngine/diagnostics.py # skip if already downloaded
python diagnostics.py
```

if it shows MinkowskiEngine not installed, check error message by running
```
python -c "import MinkowskiEngine"
```

## Troubleshooting

If encounter issues, clean up any old builds before rebuillding
```
python setup.py clean --all
rm -rf build/ dist/ *.egg-info
```

### CalledProcessError
If you encounter a `CalledProcessError` such as 
```
subprocess.CalledProcessError: Command '['ninja', '-v', '-j', '12']' returned non-zero exit status 1.
```

Refer to this related fix: https://github.com/NVlabs/instant-ngp/issues/119#issuecomment-1034701258

Or go to section [Expected initializer before ‘__s128’](https://github.com/CiSong10/MinkowskiEngine/edit/cuda12-installation/installation_note.md#expected-initializer-before-__s128)

### Undefined symbol

```
ImportError: .../MinkowskiEngine-0.5.4-py3.8-linux-x86_64.egg/MinkowskiEngineBackend/_C.cpython-38-x86_64-linux-gnu.so: undefined symbol: _ZN9minkowski8cpu_gemmIdEEv12CBLAS_LAYOUT15CBLAS_TRANSPOSES2_iiiT_PKS3_S5_S3_PS3_
```

I am not sure the root cause, but I solved it by installing dependencies and building with the flag explicitly

```
sudo apt update
sudo apt install -y build-essential cmake libopenblas-dev
CXX=g++-12 CC=gcc-12 python setup.py install --blas=openblas 
```


### Unsupported GNU Version

GCC stands for GNU Compiler Collection — 
it’s the standard C/C++ compiler used to compile most system-level and scientific code on Linux. 
Refer to [Verify GCC Version](https://github.com/CiSong10/MinkowskiEngine/edit/cuda12-installation/installation_note.md#verify-gcc-version)

### More than one instance of overloaded function "std::__to_address"

Refer to https://github.com/NVIDIA/MinkowskiEngine/issues/596

### NVTX3 header collisions 

```
.../MinkowskiEngine/src/3rdparty/cudf/detail/nvtx/nvtx3.hpp(1797): error: "event_attributes" is ambiguous
```

Refer to https://github.com/NVIDIA/MinkowskiEngine/issues/614#issuecomment-2886009673 . I have already applied the fix into my branch.


### `/user/bin/ld` cannot find 

```
/usr/bin/ld: cannot find -lcudart: No such file or directory
collect2: error: ld returned 1 exit status
```

This means the linker can'tfind the CUDA Runtime library (`libcudart.so`). You need to tell the linker where to find the CUDA libraries. 

1. Locate libcudart.so
```
find /usr/local -name "libcudart.so" 2>/dev/null
```
Typical paths: `/usr/local/cuda/lib64/libcudart.so`

2. Export environmental variables
Assume it's in /usr/local/cuda/lib64. Add these to your environment before building:
```
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
```

### Expected initializer before ‘__s128’

When building MinkowskiEngine with CUDA 11.1

**Error Message**

```
FAILED: /home/<username>/MinkowskiEngine/build/temp.linux-x86_64-cpython-38/home/<username>/MinkowskiEngine/src/math_functions_gpu.o
/usr/local/cuda-11/bin/nvcc  -I/home/cisong/miniforge3/envs/ex1/lib/python3.8/site-packages/torch/include -I/home/cisong/miniforge3/envs/ex1/lib/python3.8/site-packages/torch/include/torch/csrc/api/include -I/home/cisong/miniforge3/envs/ex1/lib/python3.8/site-packages/torch/include/TH -I/home/cisong/miniforge3/envs/ex1/lib/python3.8/site-packages/torch/include/THC -I/usr/local/cuda-11/include -I/home/cisong/MinkowskiEngine/src -I/home/cisong/MinkowskiEngine/src/3rdparty -I/home/cisong/miniforge3/envs/ex1/include/python3.8 -c -c /home/cisong/MinkowskiEngine/src/math_functions_gpu.cu -o /home/cisong/MinkowskiEngine/build/temp.linux-x86_64-cpython-38/home/cisong/MinkowskiEngine/src/math_functions_gpu.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' --expt-relaxed-constexpr --expt-extended-lambda -O3 -Xcompiler=-fno-gnu-unique -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=1 -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_86,code=sm_86 -std=c++14
/usr/include/linux/types.h:12:27: error: expected initializer before ‘__s128’
   12 | typedef __signed__ __int128 __s128 __attribute__((aligned(16)));
      |                           ^~~~~~
```

**Cause of Issue** : Linux Kernel Version incapatibility. 

CUDA 11.1 compiles better with GCC-10 and is more compatible with Linux kernel version 5.4. Therefore, you need to re-download the Linux 5.4 kernel header files and provide their path to nvcc.
(You can use `uname -r` to check your Linux Kernel version)

**Solution**

```
cd /usr/local/src
sudo wget https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-5.4.tar.xz
tar -xf linux-5.4.tar.xz
cd linux-5.4
make headers_install INSTALL_HDR_PATH=/usr/local/linux-headers-5.4
```

Then edit the `setup.py` as this [branch](https://github.com/CiSong10/MinkowskiEngine/tree/linux-kernel-compatibility)

**Reference**
* https://www.cnblogs.com/liniganma/p/18608149
* https://github.com/Pang-Yatian/Point-MAE/pull/64


## Example Configuration

### My Configuration - CUDA 12.2

```
==========System==========
Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.39
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=24.04
DISTRIB_CODENAME=noble
DISTRIB_DESCRIPTION="Ubuntu 24.04.1 LTS"
3.9.22 | packaged by conda-forge | (main, Apr 14 2025, 23:35:59)
[GCC 13.3.0]
==========Pytorch==========
2.4.0
torch.cuda.is_available(): True
==========NVIDIA-SMI==========
/usr/bin/nvidia-smi
Driver Version 552.74
CUDA Version 12.4
VBIOS Version 95.04.3c.40.84
Image Version G002.0000.00.03
GSP Firmware Version N/A
==========NVCC==========
/home/cisong/miniforge3/envs/forainet/bin/nvcc
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Feb__7_19:32:13_PST_2023
Cuda compilation tools, release 12.1, V12.1.66
Build cuda_12.1.r12.1/compiler.32415258_0
==========CC==========
CC=g++-12
/usr/bin/g++-12
g++-12 (Ubuntu 12.3.0-17ubuntu1) 12.3.0
Copyright (C) 2022 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

==========MinkowskiEngine==========
0.5.4
MinkowskiEngine compiled with CUDA Support: True
NVCC version MinkowskiEngine is compiled: 12020
CUDART version MinkowskiEngine is compiled: 12020
```


### My Configuration - CUDA 11.1

```
==========System==========
Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.10
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=24.04
DISTRIB_CODENAME=noble
DISTRIB_DESCRIPTION="Ubuntu 24.04.1 LTS"
3.8.19 | packaged by conda-forge | (default, Mar 20 2024, 12:47:35)
[GCC 12.3.0]
==========Pytorch==========
1.9.0
torch.cuda.is_available(): True
==========NVIDIA-SMI==========
/usr/lib/wsl/lib/nvidia-smi
Driver Version 552.74
CUDA Version 12.4
VBIOS Version 95.04.3c.40.84
Image Version G002.0000.00.03
GSP Firmware Version N/A
==========NVCC==========
/usr/local/cuda/bin/nvcc
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Mon_Oct_12_20:09:46_PDT_2020
Cuda compilation tools, release 11.1, V11.1.105
Build cuda_11.1.TC455_06.29190527_0
==========CC==========
/usr/bin/c++
c++ (Ubuntu 10.5.0-4ubuntu2) 10.5.0
Copyright (C) 2020 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

==========MinkowskiEngine==========
0.5.4
MinkowskiEngine compiled with CUDA Support: True
NVCC version MinkowskiEngine is compiled: 11010
CUDART version MinkowskiEngine is compiled: 11010
```
