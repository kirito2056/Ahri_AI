import torch
print(f"mps build {torch.backends.mps.is_built()}")
print(f"mps {torch.backends.mps.is_available()}")