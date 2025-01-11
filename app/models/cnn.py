import torch.nn as nn

"""
Stage 1 - Foundations
In this stage, you'll learn the basic building blocks of PyTorch. 
This includes understanding tensors, which are the fundamental data structure in PyTorch, 
similar to how arrays are fundamental in NumPy. You'll learn about tensor operations, 
autograd (automatic differentiation), and basic neural network components. 
The key topics include tensor creation, manipulation, mathematical operations, gradients, 
and how PyTorch tracks computational graphs.
"""


class BasicCNN(nn.Module):
    def __init__(self):
        self.w00 = 10
        super().__init__()
