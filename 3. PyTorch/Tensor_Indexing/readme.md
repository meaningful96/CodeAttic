# Tensor Indexing

When slicing or indexing a tensor, you can ***specify multiple rows or columns at once using another tensor***. For example, let x be represented as a (n, 1) dimensional tensor, with each element serving as an index.

In other words, if the set of indices exists as a list like [1, 2, 3, 4, 5], then x = torch.tensor([1, 2, 3, 4, 5]). Let y be the target tensor that you want to index or slice.

In this case, to specify multiple rows or columns of y at once, you can use y[x, :] or y[:, x].
