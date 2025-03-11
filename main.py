import argparse
from src.train import train_model  # ✅ 올바른 함수 호출

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model (CNN or MLP) for weight prediction.")

    # Model selection
    parser.add_argument("--model", type=str, choices=["cnn", "mlp", "linear"], required=True, help="Choose among 'cnn', 'mlp' or 'linear' model.")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")

    # Paths
    parser.add_argument("--data_path", type=str, default="./data/SOCR-HeightWeight.csv", help="Path to the dataset CSV file")
    parser.add_argument("--save_state_dict", type=str, default="./models", help="Directory to save the trained model")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for TensorBoard logs")

    args = parser.parse_args()

    # ✅ `train_model()`을 직접 호출하도록 수정
    train_model(model_type=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                data_path=args.data_path,
                save_state_dict=args.save_state_dict,
                log_dir=args.log_dir)
