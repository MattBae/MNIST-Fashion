# Standard library imports
import os
import gzip
# NumPy for numerical operations
import numpy as np
# PyTorch for neural network and data manipulation
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
# torchvision for data transformations
import torchvision.transforms as transforms
# Pillow for image processing (though PIL is generally included in Pillow)
from PIL import Image
# Scikit-learn for splitting datasets into training and validation
from sklearn.model_selection import train_test_split

# Define file paths (you can adjust these paths if necessary)
TRAIN_IMAGES_FILE = './data/train-images-idx3-ubyte.gz'
TRAIN_LABELS_FILE = './data/train-labels-idx1-ubyte.gz'
TEST_IMAGES_FILE = './data/t10k-images-idx3-ubyte.gz'
TEST_LABELS_FILE = './data/t10k-labels-idx1-ubyte.gz'

# Define the transformation
transform = transforms.Compose([
    transforms.ToPILImage(),  # Converts tensor to PIL image (necessary for transforms)
    transforms.ToTensor(),    # Converts PIL image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize images with mean=0.5 and std=0.5
])

def load_mnist_images(file_path):
    """Loads MNIST/FashionMNIST image data from .gz file"""
    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)  # Skip metadata header
    return data.reshape(-1, 28, 28)  # Reshape to (num_samples, height, width)


def load_mnist_labels(file_path):
    """Loads MNIST/FashionMNIST label data from .gz file"""
    with gzip.open(file_path, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)  # Skip metadata header
    return labels


class FashionMNISTDataset(Dataset):
    """Custom Dataset for FashionMNIST"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)  # Apply transformation

        return image, label


def get_dataloader(batch_size=32, val_size=0.2):
    """Returns DataLoader for train, validation, and test sets"""
    
    # Load the full training set
    train_images = load_mnist_images(TRAIN_IMAGES_FILE)
    train_labels = load_mnist_labels(TRAIN_LABELS_FILE)

    # Split the training data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=val_size, random_state=42
    )
    
    # Create datasets for train, validation, and test sets
    train_dataset = FashionMNISTDataset(train_images, train_labels, transform=transform)
    val_dataset = FashionMNISTDataset(val_images, val_labels, transform=transform)
    test_dataset = FashionMNISTDataset(TEST_IMAGES_FILE, TEST_LABELS_FILE, transform=transform)

    # Create DataLoader for each dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"📊 Dataset Loaded: {len(train_dataset)} training samples, {len(val_dataset)} validation samples, {len(test_dataset)} test samples")

    return train_loader, val_loader, test_loader
