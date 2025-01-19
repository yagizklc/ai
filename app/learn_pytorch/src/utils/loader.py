from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from src.config import config


def get_data():
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor(),
    )

    return (
        DataLoader(training_data, batch_size=config.batch_size),
        DataLoader(test_data, batch_size=config.batch_size),
    )
