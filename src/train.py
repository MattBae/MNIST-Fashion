import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train_model(model, train_loader, device, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    writer = SummaryWriter(log_dir=args.log_dir)

    model.train()

    # Print dataset size before training
    print(f"🟢 Starting Training: {len(train_loader.dataset)} training samples, Batch size: {args.batch_size}")

    for epoch in range(args.epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_loss, epoch+1)
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), args.save_model)
    print(f"✅ Model saved at {args.save_model}")
