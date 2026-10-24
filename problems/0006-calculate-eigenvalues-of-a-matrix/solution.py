import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    # Your implementation here
    res = torch.linalg.eigvals(matrix)
    real_nums = []
    for c in res:
        real_nums.append(float(c.real))
    real_nums.sort()
    real_nums.reverse()
    return torch.tensor(real_nums, dtype = float)
