import tvm.relay
import numpy as np
from tvm.relay.testing import mlp
from tvm.runtime import profiler_vm

target = "llvm"
dev = tvm.cpu()
mod, params = mlp.get_workload(1)
#rint(params.shape)
print(type(params))

exe = tvm.relay.vm.compile(mod, target, params=params)
vm = profiler_vm.VirtualMachineProfiler(exe, dev)

data = tvm.nd.array(np.random.rand(1, 1, 28, 28).astype("float32"), device=dev)
report = vm.profile(
    data,
    func_name="main",
    collectors=[tvm.runtime.profiling.PAPIMetricCollector()],
)
print(report)


# ######################################################################
# # Command line arguments
# import argparse
# parser = argparse.ArgumentParser(description='TVM-based frontend for FPGA deployment')
# parser.add_argument('--input',metavar='NAME', type=str, nargs='?', default='', help='file name(.pb/.onnx)')
# parser.add_argument('--input_shape',metavar='SHAPE', type=int, nargs='+', default='', help='i.e. 1 224 224 3')
# args = parser.parse_args()
# mpath = args.input
# input_shape = tuple(args.input_shape)
# print(input_shape)

# import tensorflow as tf
# shape_dict = {'input': input_shape}
# with tf.gfile.FastGFile(mpath, 'rb') as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
# sym, params = tvm.relay.frontend.from_tensorflow(graph_def, layout='NHWC', shape=shape_dict)

# target = "llvm"
# dev = tvm.cpu()

# exe = tvm.relay.vm.compile(sym, target, params=params)
# vm = profiler_vm.VirtualMachineProfiler(exe, dev)

# data = tvm.nd.array(np.random.rand(1, 416, 416, 3).astype("float32"), device=dev)
# report = vm.profile(
#     data,
#     func_name="main",
#     collectors=[tvm.runtime.profiling.PAPIMetricCollector()],
# )
# print(report)