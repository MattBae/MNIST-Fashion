import os
import gzip
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

# Define dataset directory
DATA_DIR = "./data"

# Ensure dataset exists; raise an error if missing
if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
    raise RuntimeError(f"Dataset not found in {DATA_DIR}. Please ensure it is downloaded.")

# Define file paths
TRAIN_IMAGES_FILE = os.path.join(DATA_DIR, "train-images-idx3-ubyte.gz")
TRAIN_LABELS_FILE = os.path.join(DATA_DIR, "train-labels-idx1-ubyte.gz")
TEST_IMAGES_FILE = os.path.join(DATA_DIR, "t10k-images-idx3-ubyte.gz")
TEST_LABELS_FILE = os.path.join(DATA_DIR, "t10k-labels-idx1-ubyte.gz")

# Define transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
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
    def __init__(self, images_file, labels_file, transform=None):
        self.images = load_mnist_images(images_file)
        self.labels = load_mnist_labels(labels_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)  # Apply transformation

        return image, label


# Create train and test datasets
train_dataset = FashionMNISTDataset(TRAIN_IMAGES_FILE, TRAIN_LABELS_FILE, transform=transform)
test_dataset = FashionMNISTDataset(TEST_IMAGES_FILE, TEST_LABELS_FILE, transform=transform)


def get_dataloader(batch_size=32):
    """Returns DataLoader for train and test sets"""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"📊 Dataset Loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")

    return train_loader, test_loader

