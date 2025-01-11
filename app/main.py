import torch

from app.config import config


def main():
    print(f'PyTorch version: {torch.__version__}')
    print(f'MPS (Metal) device available: {torch.backends.mps.is_available()}')
    print(f'Using device: {config.device}')


if __name__ == '__main__':
    main()
