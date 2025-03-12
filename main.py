import argparse
import sys
import os
import torch

# Ensure src directory is in the import path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import functions from src/
from train import train_model
from evaluate import evaluate_model
from dataset import get_dataloader
from model import get_model

def main():
    parser = argparse.ArgumentParser(description="Train or Evaluate a model on Fashion MNIST")

    # Model Selection
    parser.add_argument("--mode", type=str, default='train', choices=["train", "eval"], help="Choose 'train' or 'eval'")
    parser.add_argument("--model", type=str, default='resnet', choices=["resnet", "vgg", "cnn20"], help="Model type")

    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")

    # Model Save/Load Paths
    parser.add_argument("--save_model", type=str, default="./models/saved_model.pth", help="Path to save trained model")
    parser.add_argument("--load_model", type=str, default="", help="Path to load model for evaluation")

    # Logging
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for TensorBoard logs")

    args = parser.parse_args()

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = get_model(args.model).to(device)

    # Load dataset (FIXED: Removed `data_path`)
    train_loader, test_loader = get_dataloader(args.batch_size)

    if args.mode == "train":
        train_model(model, train_loader, device, args)
    elif args.mode == "eval":
        if args.load_model == "":
            print("❌ Please provide --load_model to evaluate a saved model.")
            exit(1)
        evaluate_model(model, test_loader, device, args)

if __name__ == "__main__":
    main()
