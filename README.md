# gpt-frontend

## Reference

[参考bert](https://tvm.apache.org/2020/07/14/bert-pytorch-tvm)

[nanoGPT video course](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## Build
```
git clone --recursive http://github.com/nineis7/gpt-frontend.git
```
采用http协议，防止连接不稳定

build with docker:

*参见[TVM docker cuda环境配置最新方案.md]一文*

build tvm
```
cd gpt-frontend/tvm
mkdir build
cp cmake/config.cmake build/
cd build
cmake ..
make -j8
```

目前进度：
- week1：实现driver_gpt.py 将gpt在tvm中运行并优化，测试性能作为benchmark
- week2-3：搭建WSL2+docker+cuda环境，实现tvm在WSL2中运行
- week4：实现gpt在WSL2中运行并优化，测试性能作为benchmark（包括fp32->16，fused mha）