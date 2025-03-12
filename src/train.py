import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train_model(model, train_loader, val_loader, test_loader, device, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    writer = SummaryWriter(log_dir=args.log_dir)

    model.train()

    # Print dataset size before training
    print(f"🟢 Starting Training: {len(train_loader.dataset)} training samples, Batch size: {args.batch_size}")

    for epoch in range(args.epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch", ncols=100)

        # Training loop
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), epoch=epoch+1)

        # Calculate average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_loss, epoch+1)
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")

        # Perform validation using val_loader
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:  # Use val_loader for validation
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch+1)
        print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")

        # Set model back to training mode for the next epoch
        model.train()

    torch.save(model.state_dict(), args.save_model)
    print(f"✅ Model saved at {args.save_model}")
