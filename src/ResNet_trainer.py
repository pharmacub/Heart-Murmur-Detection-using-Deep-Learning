
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, resnet34
from model_dataloader import get_loaders
from tqdm import tqdm

def _make_resnet(model_type: str, num_classes: int, in_channels: int = 1, pretrained: bool = True):
    model_type = model_type.lower()
    if model_type == "resnet18":
        model = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    elif model_type == "resnet34":
        model = resnet34(weights="IMAGENET1K_V1" if pretrained else None)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Adjust first conv for single-channel
    if in_channels != 3:
        old = model.conv1
        model.conv1 = nn.Conv2d(in_channels, old.out_channels,
                                kernel_size=old.kernel_size,
                                stride=old.stride,
                                padding=old.padding,
                                bias=False)
        if pretrained and hasattr(old, "weight") and old.weight.data.shape[1] == 3:
            with torch.no_grad():
                model.conv1.weight[:] = old.weight.mean(dim=1, keepdim=True)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def _evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            loss_sum += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return (loss_sum / max(1, len(loader))), (correct / max(1, total))

def train_resnet_model(
    model_type,
    train_csv,
    val_csv,
    model_path,
    batch_size=32,
    epochs=5,
    num_classes=3,
    in_channels=1,
    lr=1e-3,
    weight_decay=1e-4,
    pretrained=True
):
    train_loader, val_loader = get_loaders(train_csv, val_csv, batch_size=batch_size)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📡 Using device: {device}")

    model = _make_resnet(model_type, num_classes=num_classes, in_channels=in_channels, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        print(f"\n🔁 Epoch {epoch + 1}/{epochs}")
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
        print(f"✅ Epoch {epoch+1}: train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        # Save checkpoint for this epoch
        epoch_path = model_path.replace(".pth", f"_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_path)
        print(f"💾 Saved epoch model to: {epoch_path}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"🌟 New best (acc={best_val_acc:.4f}) → {model_path}")

    print(f"🏁 Training done. Best val acc: {best_val_acc:.4f}")
    return model
