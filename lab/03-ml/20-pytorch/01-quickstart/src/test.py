import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# Download training/testing data from open datasets.
def download_data():
    training_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor,
    )

    test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor(),
    )


def main():
    print(torch.__version__)
    download_data()

main()