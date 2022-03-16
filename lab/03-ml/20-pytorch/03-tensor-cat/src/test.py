import torch

data = [[1, 2], [3, 4]]
tensor = torch.tensor(data)
print(f"tensor = {tensor}")
print(f"First row: {tensor[0]}")
print(f"Second row: {tensor[1]}")

tensor2 = torch.tensor([[5, 6], [7, 8]])
t1 = torch.cat([tensor, tensor2])
print(f"t1 = {t1}")
print(f"First row of t1: {t1[0]}")
print(f"Second row of t1: {t1[1]}")
print(f"Third row of t1: {t1[2]}")