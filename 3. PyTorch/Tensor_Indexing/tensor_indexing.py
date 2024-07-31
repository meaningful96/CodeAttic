import torch
indices = torch.tensor([1,3,5,7,8,6])
print(indices)

res = torch.zeros([10,10])
res[indices, :] = 1
print(res)
