import tvm.relay
import torch
import numpy as np
import os
from datasets import load_dataset
import time
from tvm.contrib import graph_executor
import sys


# Import required libraries
import torch
from transformers import AutoTokenizer, OpenAIGPTModel

##################  prepare dataset
from datasets import load_dataset
raw_datasets = load_dataset("Imdb_small")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], max_length=512, padding="max_length", truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
from torch.utils.data import DataLoader
BATCH_SIZE = 16
eval_dataloader = DataLoader(small_eval_dataset, batch_size=BATCH_SIZE)
for batch in eval_dataloader:
    indexed_tokens = [_.numpy().tolist() for _ in batch['input_ids']]
    indexed_tokens = np.array(indexed_tokens).T.tolist()
    tokens_tensor = torch.tensor(indexed_tokens)
    print(tokens_tensor.shape)
    break

##################  model from huggineface
model = OpenAIGPTModel.from_pretrained("openai-gpt", return_dict=False)
model.eval() # Set the model in evaluation mode to deactivate the DropOut modules
for p in model.parameters():
    p.requires_grad_(False)

traced_model = torch.jit.trace(model, [tokens_tensor], strict=False)
traced_model.eval()
for p in traced_model.parameters():
    p.requires_grad_(False)

shape_list = [(i.debugName().split('.')[0], i.type().sizes()) for i in  list(traced_model.graph.inputs())[1:]]
mod, params = tvm.relay.frontend.pytorch.from_pytorch(traced_model, shape_list, default_dtype="float32")
# print(type(params))
# mod = tvm.relay.transform.InferType()(mod)

# from tvm.relay.testing import mlp
from tvm.runtime import profiler_vm

target = "cuda"
dev = tvm.cuda(0)

exe = tvm.relay.vm.compile(mod, target, params=params)
vm = profiler_vm.VirtualMachineProfiler(exe, dev)

data = tvm.nd.array(np.random.rand(16, 512).astype(int), device=dev)
report = vm.profile(
    data,
    func_name="main",
    collectors=[tvm.runtime.profiling.PAPIMetricCollector()],
)

import sys
sys.stdout = open('PAPI_profiling_gpt_cuda.txt', mode = 'w',encoding='utf-8')
print(report)