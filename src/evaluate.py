import torch
import argparse
import os
from model import CNNRegressor
from dataset import load_data

import logging

# Setup logging
log_file = "./logs/evaluation_results.txt"
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.DEBUG, format="%(asctime)s - %(message)s", filemode="a")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained CNN model.")
    
    parser.add_argument("--data_path", type=str, default="./data/SOCR-HeightWeight.csv",
                        help="Path to the dataset CSV file")
    parser.add_argument("--save_state_dict", type=str, default="./models",
                        help="Directory where the trained model is stored")

    args = parser.parse_args()

    # Load Data
    logging.info(f"📂 Loading dataset from: {args.data_path}")
    _, _, test_loader = load_data(args.data_path)

    # Load Model
    model_path = os.path.join(args.save_state_dict, "cnn_regressor.pth")
    logging.info(f"💾 Loading model from: {model_path}")
    
    model = CNNRegressor()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    logging.info(f"✅ Model loaded successfully! Beginning evaluation...")

    # Evaluate
    test_loss = 0
    with torch.no_grad():
        for X, Y in test_loader:
            output = model(X)
            test_loss += ((output - Y) ** 2).sum().item()
    
    avg_test_loss = test_loss / len(test_loader)

    # Log the test results
    result_message = f"📊 Final Test Loss: {avg_test_loss:.4f}"
    print(result_message)
    logging.info(result_message)
