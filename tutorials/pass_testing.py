import numpy as np
import tvm
import tvm.relay as relay


############################## Use tvm/python/tvm/realy/testing module mlp
from tvm.relay.testing import mlp

mod, params = mlp.get_workload(1)
print(mod)
print(params)