# gpt-frontend

## Reference

[参考bert](https://tvm.apache.org/2020/07/14/bert-pytorch-tvm)

[nanoGPT video course](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## Build
```
git clone --recursive http://github.com/nineis7/gpt-frontend.git
```
采用http协议，防止连接不稳定

#### build with docker:

*参见[TVM docker cuda环境配置最新方案.md]一文*

#### build tvm
```
cd gpt-frontend/tvm
mkdir build
cp cmake/config.cmake build/
cd build
cmake ..
make -j8
```

#### build with PAPI (cmake has been updated to 3.24.1 in cmake_source.sh)
```
# Method 1：在docker/install中将ubuntu_install_papi.sh添加进build.sh中
# 需要修改export PAPI_CUDA_ROOT=/usr/local/cuda 为cuda->该文件夹下cuda版本号

# Method 2：手动安装
git clone --branch papi-6-0-0-1-t https://bitbucket.org/icl/papi.git
cd papi/src
export PAPI_CUDA_ROOT=/usr/local/cuda版本号（需自行查看）
# export PAPI_ROCM_ROOT=/opt/rocm 可以不安装
./configure --with-components="cuda"
make && make install

安装后重新cmake|make来build tvm
```

## 目前进度：
- week1：实现driver_gpt.py 将gpt在tvm中运行并优化，测试性能作为benchmark
- week2-3：搭建WSL2+docker+cuda环境，实现tvm在WSL2中运行
- week4：实现gpt在WSL2中运行并优化，测试性能作为benchmark（包括fp32->16，fused mha）
- week5；PAPI编译安装，进行基于llvm与cuda的gpt model性能测试，结果见artifacts/PAPI_profiling