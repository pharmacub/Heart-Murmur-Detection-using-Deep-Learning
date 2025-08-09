
import os
import torch
import torch.nn as nn
import torch.optim as optim
from model_dataloader import get_loaders
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import numpy as np
import pandas as pd

class LSTMSoundClassifier(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, num_layers=2, num_classes=3, dropout=0.3):
        super(LSTMSoundClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # x: [B, 1, 128, T] -> reshape to [B, T, 128]
        B, C, F, T = x.shape
        x = x.view(B, F, T).permute(0, 2, 1)  # [B, T, F]
        out, _ = self.lstm(x)                  # [B, T, 2H]
        out = out[:, -1, :]                    # last timestep
        return self.fc(out)

def _compute_class_weights(csv_path):
    df = pd.read_csv(csv_path)
    label_map = {"Absent": 0, "Present": 1, "Unknown": 2}
    y = df["murmur_label"].map(label_map).values
    classes = np.array([0,1,2])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return torch.tensor(weights, dtype=torch.float32)

def _evaluate(model, loader, device, criterion):
    model.eval()
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

def train_lstm_model(train_csv, val_csv, model_dir, batch_size=32, epochs=10, lr=1e-3, weight_decay=1e-4):
    train_loader, val_loader = get_loaders(train_csv, val_csv, batch_size=batch_size)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📡 Using device: {device}")

    model = LSTMSoundClassifier(num_classes=3).to(device)

    # optional class weights
    try:
        class_weights = _compute_class_weights(train_csv).to(device)
        print(f"⚖️ Using class weights: {class_weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    except Exception as e:
        print(f"⚠️ Could not compute class weights: {e}. Using unweighted CE.")
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    os.makedirs(model_dir, exist_ok=True)
    best_val_acc = 0.0
    best_path = os.path.join(model_dir, "lstm_best.pth")

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        print(f"\n🔁 Epoch {epoch}/{epochs}")
        for X, y in tqdm(train_loader, desc="Training", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))
        val_loss, val_acc = _evaluate(model, val_loader, device, criterion)
        scheduler.step(val_loss)
        print(f"✅ Epoch {epoch}: train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        # Save per-epoch
        ep_path = os.path.join(model_dir, f"lstm_epoch{epoch}.pth")
        torch.save(model.state_dict(), ep_path)
        print(f"💾 Saved: {ep_path}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"🌟 New best (acc={best_val_acc:.4f}) → {best_path}")

    print(f"🏁 Training complete. Best val acc: {best_val_acc:.4f}")
    return model
