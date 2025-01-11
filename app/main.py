import torch

from app.config import config
from app.utils.loader import get_data


def main():
    print(f'PyTorch version: {torch.__version__}')
    print(f'MPS (Metal) device available: {torch.backends.mps.is_available()}')
    print(f'Using device: {config.device}')

    train_data, test_data = get_data()
    for x, y in train_data:
        print(f'Shape of X [N, C, H, W]: {x.shape}')
        print(f'Shape of y {y.shape} {y.dtype}')


if __name__ == '__main__':
    main()
