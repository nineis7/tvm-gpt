import numpy as np
import tvm
from tvm import te
import tvm.relay as relay

###############################################################################
# Create An Example Relay Program
# -------------------------------
# First of all, we create a simple Relay program for the tutorial. This program
# will be used by various optimizations of the examples in this tutorial.
# Similarly, users can write a tir primitive function and apply the tir passes.


def example():
    shape = (1, 64, 54, 54)
    c_data = np.empty(shape).astype("float32")
    c = relay.const(c_data)
    weight = relay.var("weight", shape=(64, 64, 3, 3))
    x = relay.var("x", relay.TensorType((1, 64, 56, 56), "float32"))
    conv = relay.nn.conv2d(x, weight)
    drop = relay.nn.dropout(conv)
    

    return relay.Function([x, weight], z2)