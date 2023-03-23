# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=line-too-long
"""
.. _tutorial-use-pass-infra:
How to Use TVM Pass Infra
=========================
**Author**: `Zhi Chen <https://github.com/zhiics>`_
随着 Relay/tir 中优化pass次数的增加，手动执行它们并维护它们的依赖关系变得棘手。 因此，我们引入了一个Pass基础设施来管理优化passes，并使其适用于 TVM 栈中不同层的 IR。 
Relay/tir 程序的优化可以应用于各种粒度，即分别使用 `tvm.relay.transform.FunctionPass`/`tvm.tir.transform.PrimFuncPass` 和 `tvm.transform.ModulePass` 的function-level
和module-level级别的优化。 或者用户可以依靠 `tvm.transform.Sequential` 在 Relay/tir 程序上应用一系列passes，其中passes之间的依赖关系可以通过Pass Infra解决。 
这里主要是来演示一些开发人员如何使用Pass Infra来进行某种优化，并为Relay程序创建优化管道。这里的方法同样适用于tir。首先导入一些必要的包。

本篇教程来自<https://tvm.hyper.ai/docs/how_to/extend/pass_infra/>
"""


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
    y = relay.add(c, c)
    y = relay.multiply(y, relay.const(2, "float32"))
    y = relay.add(conv, y)
    z = relay.add(y, c)
    z1 = relay.add(y, c)
    z2 = relay.add(z, z1)
    return relay.Function([x, weight], z2)


###############################################################################
# Let us register layout alteration for a conv2d op so that we can apply the
# layout alteration pass on the example. How alter layout pass works is out
# the scope of this tutorial.


@relay.op.register_alter_op_layout("nn.conv2d", level=101)
def alter_conv2d(attrs, inputs, tinfos, out_type):
    data, weight = inputs
    new_attrs = dict(attrs)
    new_attrs["data_layout"] = "NCHW16c"
    return relay.nn.conv2d(data, weight, **new_attrs)


###############################################################################
# Optimize the Program
# --------------------
# Now we would like to optimize the program. Relay features a host of
# optimizations. We will select some of them to apply on this example program.
#
# There are multiple ways to optimize a Relay program. Below we will provide
# examples for each of them.
#
# Manually Apply Optimization Passes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Let's first create a relay Module which contains one or multiple Relay
# functions for optimization.
f = example()
mod = tvm.IRModule.from_expr(f)

print(mod)
'''
其中meta[relay.Constant][0]即c

def @main(%x: Tensor[(1, 64, 56, 56), float32], %weight: Tensor[(64, 64, 3, 3), float32]) {
  %0 = add(meta[relay.Constant][0], meta[relay.Constant][0]);
  %1 = nn.conv2d(%x, %weight, padding=[0, 0, 0, 0]);
  %2 = multiply(%0, 2f);
  %3 = add(%1, %2);
  %4 = add(%3, meta[relay.Constant][0]);
  %5 = add(%3, meta[relay.Constant][0]);
  add(%4, %5)
}
'''

# Now we can apply constant folding on the module.
# fold_const here is a callback that doesn't take any parameters.
fold_const = relay.transform.FoldConstant()
# Then, we can invoke the pass on the given module. Note that the constant
# folding pass works at the function-level. That being said, each function in
# the module will be applied with the optimization. Users don't need to iterate
# through individual functions manually to apply this pass.
mod = fold_const(mod)
# We can see from the updated program that the constants are folded.
print(mod)
'''
常量折叠pass中将relay里 
    %0 = add(meta[relay.Constant][0], meta[relay.Constant][0]);
    %2 = multiply(%0, 2f);
两句直接合并为meta[relay.Constant][0]
将c作为meta[relay.Constant][1]存放

def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %0 = nn.conv2d(%x, %weight, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %1 = add(%0, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %2 = add(%1, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %3 = add(%1, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  add(%2, %3) /* ty=Tensor[(1, 64, 54, 54), float32] */
}
'''

###############################################################################
# More optimizations can be applied in the similar manner. For instance, we can
# eliminate the common expressions that used by `z` and `z1`.
mod = relay.transform.EliminateCommonSubexpr()(mod)
print(mod)
'''
在expr中，有
    z = relay.add(y, c)
    z1 = relay.add(y, c)
两者的右值计算式相同，故可以进行公共子表达式消除，消除z1一句，在add中直接将z和z相加

def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %0 = nn.conv2d(%x, %weight, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %1 = add(%0, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %2 = add(%1, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  add(%2, %2) /* ty=Tensor[(1, 64, 54, 54), float32] */
}
'''

###############################################################################
# Some optimizations, such as fusion, are parameteric as well. For example,
# opt level 0 will not allow operators to be fused together. Users can pass the
# `fuse_opt_level` to enable this.
mod = relay.transform.FuseOps(fuse_opt_level=0)(mod)

# We can observe that the optimized module contains functions that only have
# a signle primitive op.
print(mod)

'''
fuseop生成新的funciton(即fn)，删去type注释后fn如下：
%0 = fn (%p03: Tensor[(1, 64, 56, 56), float32], 
         %p12: Tensor[(64, 64, 3, 3), float32], Primitive=1) -> Tensor[(1, 64, 54, 54), float32] 
        { nn.conv2d(%p03, %p12, padding=[0, 0, 0, 0]) }
%1 = %0(%x, %weight)
其中 %p03 作为 %x 的形参（Tensor[(1, 64, 56, 56)），%p12 作为 %weight 的形参（Tensor[(64, 64, 3, 3)）
而此时nn.conv2d不再作为relay中的node，而是作为%0 fn的内部计算的一个步骤

fuse_opt_level=0时，每个fn只包含一个op，相当于没有进行任何fuse操作，只是包装了一层fnnode，
这一层包装相当于在relay中具体的op的node全部被fn node替代，整个relay全是fn node而不存在游荡的primitive node，
这也正是fuse所作的，将relay中的各个primitive node以合理的方式替换为更high-level的fn node

def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %0 = fn (%p03: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %p12: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    nn.conv2d(%p03, %p12, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 56, 56), float32], Tensor[(64, 64, 3, 3), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %1 = %0(%x, %weight) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %2 = fn (%p02: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, %p11: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    add(%p02, %p11) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 54, 54), float32], Tensor[(1, 64, 54, 54), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %3 = %2(%1, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %4 = fn (%p01: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, %p1: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    add(%p01, %p1) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 54, 54), float32], Tensor[(1, 64, 54, 54), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %5 = %4(%3, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %6 = fn (%p0: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    add(%p0, %p0) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 54, 54), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %6(%5) /* ty=Tensor[(1, 64, 54, 54), float32] */
}
'''

###############################################################################
# Use Sequential to Apply a Sequence of Passes
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Applying passes as above is actually tedious and it may require users to have
# better understanding about the dependencies between them. For example, fusion
# currently doesn't work well on let bindings. Therefore, we would not be able
# to fuse operators that were fusable if :py:func:`relay.transform.ToANormalForm` is applied before
# fusion, as this pass generates let bindings for each expression to
# canonicalize a Relay program.
#
# Relay, hence, provides :py:class:`tvm.transform.Sequential` to alleviate developers from handling
# these issues explicitly by specifying the required passes of each pass and
# packing them as a whole to execute. For example, the same passes can now be
# applied using the sequential style as the following. :py:class:`tvm.transform.Sequential` is
# similiar to `torch.nn.sequential <https://pytorch.org/docs/stable/nn.html#torch.nn.Sequential>`_
# and `mxnet.gluon.block <https://mxnet.apache.org/api/python/docs/_modules/mxnet/gluon/block.html>`_.
# For example, `torch.nn.sequential` is used to contain a sequence of PyTorch
# `Modules` that will be added to build a network. It focuses on the network
# layers. Instead, the :py:class:`tvm.transform.Sequential` in our pass infra works on the optimizing
# pass.

# Now let's execute some passes through :py:class:`tvm.transform.Sequential`
f = example()
mod = tvm.IRModule.from_expr(f)
# Glob the interested passes.
seq = tvm.transform.Sequential(
    [
        relay.transform.FoldConstant(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.FuseOps(fuse_opt_level=2),
    ]
)
mod1 = seq(mod)
print(mod1)
'''
将各个transform的pass用sequential包起来，便无需考虑内部pass之间顺序导致的异常情况了(但仍然出现异常)
异常情况见[artifacts/Records/pass_q.pptx]，添加squential机制并没有解决问题
fuse_opt_level=2 的fuse将expr中所有op均fuse到了一起，但squential中并没有执行EliminateCommonSubexpr() Pass，
原因是level不够

def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %4 = fn (%p0: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %p1: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */, %p2: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, %p3: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    %0 = nn.conv2d(%p0, %p1, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    %1 = add(%0, %p2) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    %2 = add(%1, %p3) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    %3 = add(%1, %p3) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    add(%2, %3) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 56, 56), float32], Tensor[(64, 64, 3, 3), float32], Tensor[(1, 64, 54, 54), float32], Tensor[(1, 64, 54, 54), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %4(%x, %weight, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */
}
'''

###############################################################################
# From the transformed Relay program, we can see that there are still two
# identical addition operations. This is because ``EliminateCommonSubexpr``
# was not actually performed. The reason is because only the passes that have
# optimization level less or equal to 2 will be executed by default under
# :py:class:`tvm.transform.Sequential`. The pass infra,
# however, provides a configuration interface
# for users to customize the optimization level that they want to execute.

with tvm.transform.PassContext(opt_level=3):
    mod2 = seq(mod)
print(mod2)
'''
def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %3 = fn (%p0: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %p1: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */, %p2: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, %p3: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    %0 = nn.conv2d(%p0, %p1, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    %1 = add(%0, %p2) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    %2 = add(%1, %p3) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    add(%2, %2) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 56, 56), float32], Tensor[(64, 64, 3, 3), float32], Tensor[(1, 64, 54, 54), float32], Tensor[(1, 64, 54, 54), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %3(%x, %weight, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */
}
'''

###############################################################################
# Now we can see that only one of the two identical additions is kept.
#
# In addition, users can selectively disable some passes using the
# `disabled_pass` config, which is similar to the `-fno-xxx` option used the
# general purpose compilers, such as Clang and GCC. For example, we can disable
# EliminateCommonSubexpr as following. The printed module will again show two
# identical addition operations.

with tvm.transform.PassContext(opt_level=3, disabled_pass=["EliminateCommonSubexpr"]):
    mod3 = seq(mod)
print(mod3)
'''
def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %4 = fn (%p0: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %p1: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */, %p2: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, %p3: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    %0 = nn.conv2d(%p0, %p1, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    %1 = add(%0, %p2) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    %2 = add(%1, %p3) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    %3 = add(%1, %p3) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    add(%2, %3) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 56, 56), float32], Tensor[(64, 64, 3, 3), float32], Tensor[(1, 64, 54, 54), float32], Tensor[(1, 64, 54, 54), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %4(%x, %weight, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */
}
'''

###############################################################################
# The passes applied so far are target independent. The pass infra also
# provides a means to make pass target-aware. For example, the layout
# alteration pass falls in such category.

with tvm.transform.PassContext(opt_level=3):
    mod4 = seq(mod)
print(mod4)
'''
def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %3 = fn (%p0: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %p1: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */, %p2: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, %p3: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    %0 = nn.conv2d(%p0, %p1, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    %1 = add(%0, %p2) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    %2 = add(%1, %p3) /* ty=Tensor[(1, 64, 54, 54), float32] */;
    add(%2, %2) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 56, 56), float32], Tensor[(64, 64, 3, 3), float32], Tensor[(1, 64, 54, 54), float32], Tensor[(1, 64, 54, 54), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %3(%x, %weight, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */
}
'''

seq1 = tvm.transform.Sequential([relay.transform.AlterOpLayout()])
with tvm.transform.PassContext(opt_level=3):
    with tvm.target.Target("llvm"):
        mod5 = seq1(mod)
print(mod5)
'''
def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %0 = layout_transform(%x, src_layout="NCHW", dst_layout="NCHW16c") /* ty=Tensor[(1, 4, 56, 56, 16), float32] */;
  %1 = add(meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %2 = multiply(%1, 2f /* ty=float32 */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %3 = nn.conv2d(%0, %weight, padding=[0, 0, 0, 0], data_layout="NCHW16c") /* ty=Tensor[(1, 4, 54, 54, 16), float32] */;
  %4 = layout_transform(%2, src_layout="NCHW", dst_layout="NCHW16c") /* ty=Tensor[(1, 4, 54, 54, 16), float32] */;
  %5 = add(%3, %4) /* ty=Tensor[(1, 4, 54, 54, 16), float32] */;
  %6 = layout_transform(meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */, src_layout="NCHW", dst_layout="NCHW16c") /* ty=Tensor[(1, 4, 54, 54, 16), float32] */;
  %7 = add(%5, %6) /* ty=Tensor[(1, 4, 54, 54, 16), float32] */;
  %8 = add(%5, %6) /* ty=Tensor[(1, 4, 54, 54, 16), float32] */;
  %9 = add(%7, %8) /* ty=Tensor[(1, 4, 54, 54, 16), float32] */;
  layout_transform(%9, src_layout="NCHW16c", dst_layout="NCHW") /* ty=Tensor[(1, 64, 54, 54), float32] */
}
'''

##############################################################################
# Implement a Pass Using Python Decorator
# ------------------------------------------
# The next example illustrates how we can orchestrate a customized optimization
# pipeline through the pass infra using Python decorators. This functionality
# greatly eases the implementation of passes. For example, users can simply
# define a decorated class to do function-level optimizations as the following
# example shows. `transform_function` wraps a class to replace all constants
# with a multiple of `c`. Later on, each function in a given module will be
# visited and each constant in the function will be replaced when we invoke the
# customized pass.


@relay.transform.function_pass(opt_level=1)
class CustomPipeline:
    """Simple test function to replace one argument to another."""

    def __init__(self, multiplier):
        self.multiplier = multiplier

    # This function can define a pass.
    def transform_function(self, func, mod, ctx):
        obj = self

        class ReplaceConstant(tvm.relay.ExprMutator):
            def visit_constant(self, c):
                return relay.multiply(obj.multiplier, c)

        return ReplaceConstant().visit(func)


f = example()
mod = tvm.IRModule.from_expr(f)
custom_pass = CustomPipeline(multiplier=relay.const(3, "float32"))
assert custom_pass.info.name == "CustomPipeline"
mod3 = custom_pass(mod)
print(mod3)
'''
def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %0 = multiply(3f /* ty=float32 */, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %1 = add(%0, %0) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %2 = multiply(3f /* ty=float32 */, 2f /* ty=float32 */) /* ty=float32 */;
  %3 = nn.conv2d(%x, %weight, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %4 = multiply(%1, %2) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %5 = add(%3, %4) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %6 = add(%5, %0) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %7 = add(%5, %0) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  add(%6, %7) /* ty=Tensor[(1, 64, 54, 54), float32] */
}
'''

##############################################################################
# Debug a Pass
# ------------
# TVM provides users a plug-and-play style debugging pass that print the IR
# after a certain pass is done through a special pass (``PrintIR``) to dump the IR of the
# whole module. A slightly modified version of the sequential pass example
# could be like the following to enable IR dumping for ``FoldConstant`` optimization.

f = example()
mod = tvm.IRModule.from_expr(f)
seq = tvm.transform.Sequential(
    [
        relay.transform.FoldConstant(),
        tvm.transform.PrintIR(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.FuseOps(),
        relay.transform.AlterOpLayout(),
    ]
)

# By inserting the ``PrintIR`` pass after ``FoldConstant``, the pass infra will
# dump out the module IR when ``FoldConstant`` is done. Users can plug in this
# pass after any pass they want to debug for viewing the optimization effect.
#
# There is a more flexible debugging mechanism also exposed by the build configuration
# object. One can pass a tracing function which can be used to execute arbitrary code
# before and/or after each pass. A tracing function will receive a :py::class:`tvm.IRModule`,
# a :py:class:`tvm.transform.PassInfo` object,
# and a boolean indicating whether you are executing before, or after a pass.
# An example is below.


@tvm.instrument.pass_instrument
class PrintIR:
    """仅在 pass 执行之前打印 pass 的名称，IR。"""

    def run_before_pass(self, mod, info):
        print("Running pass: {}", info)
        print(mod)

with tvm.transform.PassContext(opt_level=3, instruments=[PrintIR()]):
    with tvm.target.Target("llvm"):
        # Perform the optimizations.
        mod = seq(mod)
print(mod)

print("done")
'''
Running pass: {} The meta data of the pass - pass name: InferType, opt_level: 0, required passes: []
    ...
'''

##############################################################################
# Summary
# -------
# This tutorial has covered how we can write and invoke passes in TVM more
# conveniently using the pass infra. Different ways of invoking a pass are also
# disucssed. Using :py:class:`tvm.transform.Sequential` can largely help
# users to ease the work of handling multiple optimization passes and their
# dependencies. In addition, an example is provided to illustrate
# how we can debug a pass using the ``PrintIR`` and tracing.