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
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
