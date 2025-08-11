
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model_dataloader import get_loaders
from cnn_attention_model import CNNAttentionClassifier
from tqdm import tqdm

def _evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return (total_loss / max(1, len(loader))), (correct / max(1, total))

def train_attention_model(train_csv, val_csv, base_model_path, batch_size=32, epochs=10, lr=1e-3, weight_decay=1e-4):
    train_loader, val_loader = get_loaders(train_csv, val_csv, batch_size=batch_size)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")

    model = CNNAttentionClassifier(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    os.makedirs(os.path.dirname(base_model_path), exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        print(f"\n Epoch {epoch + 1}/{epochs}")
        for X, y in tqdm(train_loader, desc="Training", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))
        val_loss, val_acc = _evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        print(f" Epoch {epoch+1}: train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        # Save epoch checkpoint
        ep_path = base_model_path.replace(".pth", f"_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), ep_path)
        print(f" Saved epoch model to: {ep_path}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), base_model_path)
            print(f" New best (acc={best_val_acc:.4f}) → {base_model_path}")

    print(f" Training done. Best val acc: {best_val_acc:.4f}")
    return model
