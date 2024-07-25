In PyTorch, squeeze and unsqueeze are tensor manipulation functions that adjust the dimensions of tensors. Here's an explanation of each function along with examples:

# torch.squeeze
The squeeze function removes all singleton dimensions (dimensions with size 1) from the tensor.

- Usage:
    - torch.squeeze(input, dim=None)
    - input: The input tensor.
    - dim: (Optional) If given, the function will squeeze only the specified dimension.

\[**Example**\]
```python
import torch

# Create a tensor with a singleton dimension
tensor = torch.randn(1, 3, 1, 4)
print("Original tensor shape:", tensor.shape)

# Squeeze the tensor
squeezed_tensor = torch.squeeze(tensor)
print("Squeezed tensor shape:", squeezed_tensor.shape)

# Squeeze a specific dimension
squeezed_tensor_dim = torch.squeeze(tensor, dim=2)
print("Squeezed tensor shape along dimension 2:", squeezed_tensor_dim.shape)
```

\[**Output**\]
```bash
Original tensor shape: torch.Size([1, 3, 1, 4])
Squeezed tensor shape: torch.Size([3, 4])
Squeezed tensor shape along dimension 2: torch.Size([1, 3, 4])
```

# torch.unsqueeze
The unsqueeze function adds a singleton dimension at the specified position.
- Usage:
    - torch.unsqueeze(input, dim)
    - input: The input tensor.
    - dim: The index at which to insert the singleton dimension.
\[**Example**\]
```python

import torch

# Create a tensor
tensor = torch.randn(3, 4)
print("Original tensor shape:", tensor.shape)

# Unsqueeze the tensor
unsqueezed_tensor = torch.unsqueeze(tensor, dim=0)
print("Unsqueezed tensor shape (at dim=0):", unsqueezed_tensor.shape)

# Unsqueeze the tensor at a different dimension
unsqueezed_tensor_dim = torch.unsqueeze(tensor, dim=2)
print("Unsqueezed tensor shape (at dim=2):", unsqueezed_tensor_dim.shape)
```

\[**Output**\]
```bash
Original tensor shape: torch.Size([3, 4])
Unsqueezed tensor shape (at dim=0): torch.Size([1, 3, 4])
Unsqueezed tensor shape (at dim=2): torch.Size([3, 4, 1])
```

These functions are useful for manipulating the shape of tensors to meet the requirements of various operations in neural network models.
