import os
import torchvision.transforms as transforms
from torchvision import datasets

# Define dataset directory
DATA_DIR = "./data/FashionMNIST"

# Check if dataset is already downloaded
if not os.path.exists(DATA_DIR):
    download_flag = True
else:
    download_flag = False

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset with proper download flag
train_dataset = datasets.FashionMNIST(root=DATA_DIR, train=True, transform=transform, download=download_flag)
test_dataset = datasets.FashionMNIST(root=DATA_DIR, train=False, transform=transform, download=download_flag)
