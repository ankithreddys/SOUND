import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor




def download_mnist():
    train_data = datasets.MNIST(
        root="data",
        download=True,
        train=True,
        transform=ToTensor( ))
    


    validation_data = datasets.MNIST(
        root="data",
        download=True,
        train=False,
        transform=ToTensor( ))
    
    return train_data, validation_data


if __name__=="__main__":
    train_data, _ = download_mnist()
    print("Dataset Downloaded")
