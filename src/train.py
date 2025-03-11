import torch
import torch.optim as optim
import torch.nn as nn
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from src.model import CNNRegressor, MLPRegressor, SimpleLinearRegressor
from src.dataset import load_data

# Logging 설정
log_file = "./logs/training_progress.txt"
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(filename=log_file, level=logging.DEBUG, format="%(asctime)s - %(message)s", filemode="a")

def train_model(model_type, epochs=100, batch_size=32, learning_rate=0.001,
                data_path="./data/SOCR-HeightWeight.csv",
                save_state_dict="./models", log_dir="./logs"):

    logging.info(f"Training {model_type} model | Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")
    
    train_loader, val_loader, _ = load_data(data_path, batch_size=batch_size)

    # 모델 선택
    if model_type == "cnn":
        model = CNNRegressor()
    elif model_type == "mlp":
        model = MLPRegressor()
    elif model_type == "linear":
        model = SimpleLinearRegressor()
    else:
        raise ValueError("Invalid model type! Choose either 'cnn', 'mlp', or 'linear'.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model_save_path = os.path.join(save_state_dict, f"{model_type}_regressor.pth")
    os.makedirs(save_state_dict, exist_ok=True)


    # log_dir을 arguments에서 받아와서 사용
    model_log_dir = os.path.join(log_dir, model_type)
    os.makedirs(model_log_dir, exist_ok=True)

    # ✅ 현재 날짜/시간 추가하여 TensorBoard 로그 디렉토리 생성 # 타임스탬프 추가
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # TensorBoard SummaryWriter 설정 (tfevents 파일 이름에 타임스탬프 추가)
    writer = SummaryWriter(log_dir=model_log_dir, filename_suffix=f"_{timestamp}")

    epoch_bar = tqdm(total=epochs, desc=f"Training {model_type.upper()}", unit="epoch", dynamic_ncols=True, leave=False)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for X, Y in train_loader:
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, Y in val_loader:
                output = model(X)
                val_loss += criterion(output, Y).item()
        avg_val_loss = val_loss / len(val_loader)

        # ✅ 같은 Loss 그래프에서 CNN, MLP, Linear Regression 비교 가능
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

        epoch_bar.set_postfix(loss=f"{avg_train_loss:.4f}", val_loss=f"{avg_val_loss:.4f}")
        epoch_bar.update(1)
        tqdm.write(f"Epoch [{epoch+1}/{epochs}] | {model_type.upper()} Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    writer.close()
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"✅ {model_type.upper()} Model saved at {model_save_path}")
