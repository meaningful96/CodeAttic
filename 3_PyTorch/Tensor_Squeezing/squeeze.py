import torch

## Squeeze
x1_origin = torch.rand(1, 1, 20, 128)
x1 = x1_origin.squeeze() # [1, 1, 20, 128] -> [20, 128]

x2_origin = torch.rand(1, 1, 20, 128)
x2 = x2_origin.squeeze(dim=1) # [1, 1, 20, 128] -> [1, 20, 128]

print("*"*80)
print("Squeezing!!")
print(f"The size of vector x1 before squeezing is: {x1_origin.size()}")
print(f"The size of vector x1 is: {x1.shape}")
print()
print(f"The size of vector x2 before squeezing is: {x2_origin.size()}")
print(f"The size of vector x2 is: {x2.shape}")
print("*"*80)


## Unsqueeze

x3_origin = torch.rand(3, 20, 128)
x3 = x3_origin.unsqueeze(dim=1) #[3, 20, 128] -> [3, 1, 20, 128]

print("*"*80)
print("Unsqueezing!!")
print(f"The size of vector x3 before squeezing is: {x3_origin.size()}")
print(f"The size of vector x3 is: {x3.shape}")
print("*"*80)


