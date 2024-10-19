from transformers import AutoModelForCausalLM 
import torch

# 检查可用CUDA设备数量
device_count = torch.cuda.device_count()
print(f"Available CUDA devices: {device_count}")
# 根据可用设备动态设置max_memory
if device_count > 0:
    kwargs = {
        "max_memory": {i: '20GiB' for i in range(device_count)},
        "device_map": "auto",  # 这里也可以直接指定设备，例如 {0: 'cuda:0'} 
        "revision": "main",
    }
else:
    kwargs = {
        "device_map": "cpu",  # 如果没有GPU，使用CPU
    }

model = AutoModelForCausalLM.from_pretrained(
            "./llama-7b",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )


Batch structure: {
    'id': [103375, 13076, 137599, 6648], 
    'edge_index': tensor([[36, 36, 36, 39,  5, ],[ 5,  8, 41, 38, 23,]]), 
    'mapping': tensor([34, 52, 70, 71]), 
    'batch': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3,
        3, 3, 3, 3])}