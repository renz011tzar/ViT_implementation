import torchvision.transforms as transforms
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import random_split

def create_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
    ])
    return transform

def load_data(path, transform):
    data = ImageFolder(root=path, transform=transform)
    return data

def create_data_loader(data, batch_size, shuffle, num_workers):
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader

def get_classes(data):
    return data.classes

def split_dataset(dataset, train_ratio):
    datasize=len(dataset)
    train_size = int(train_ratio * len(dataset))  
    test_size = len(dataset) - train_size  

    trainset, testset = random_split(dataset, [train_size, test_size])
    return trainset, testset, datasize

class CustomSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices):
        super(CustomSubset, self).__init__(dataset, indices)
        self.classes = dataset.classes

