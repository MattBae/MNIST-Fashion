import torch
from torch.utils.tensorboard import SummaryWriter

def evaluate_model(model, test_loader, device, args):
    model.load_state_dict(torch.load(args.load_model))
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=args.log_dir)

    # Print dataset size before evaluation
    print(f"🔵 Starting Evaluation: {len(test_loader.dataset)} test samples")

    total_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    writer.add_scalar("Loss/Validation", avg_loss)
    print(f"✅ Evaluation Completed. Validation Loss: {avg_loss:.4f}")
