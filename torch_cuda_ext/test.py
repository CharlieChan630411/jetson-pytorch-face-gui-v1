import torch
import add_cuda

x = torch.ones(10, device="cuda")
y = torch.ones(10, device="cuda")
out = torch.zeros(10, device="cuda")

add_cuda.add(x, y, out)
print(out)

