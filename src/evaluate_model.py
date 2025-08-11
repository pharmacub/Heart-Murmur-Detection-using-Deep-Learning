
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
import pandas as pd

from model_dataloader import MyHeartDataset

def _make_resnet(model_type: str, num_classes: int, in_channels: int = 1):
    model_type = model_type.lower()
    if model_type == "resnet18":
        model = resnet18(weights=None)
    elif model_type == "resnet34":
        model = resnet34(weights=None)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    if in_channels != 3:
        old = model.conv1
        model.conv1 = nn.Conv2d(in_channels, old.out_channels,
                                kernel_size=old.kernel_size,
                                stride=old.stride,
                                padding=old.padding,
                                bias=False)
        with torch.no_grad():
            if old.weight.data.shape[1] == 3:
                model.conv1.weight[:] = old.weight.mean(dim=1, keepdim=True)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def evaluate_on_test(model_path, test_csv, model_name, model_type="resnet18", batch_size=32, in_channels=1, num_classes=3):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")

    test_ds = MyHeartDataset(test_csv)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = _make_resnet(model_type, num_classes=num_classes, in_channels=in_channels).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    avg_loss = total_loss / max(1, len(test_loader))
    report = classification_report(all_labels, all_preds, target_names=["Absent", "Present", "Unknown"], digits=4)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    print("\n Classification Report:\n" + report)
    print(f" Avg Test Loss: {avg_loss:.4f} |  Macro F1: {macro_f1:.4f}")

    # Save CSV row
    results_row = {
        "model_name": model_name,
        "model_type": model_type,
        "test_loss": avg_loss,
        "macro_f1": macro_f1,
    }
    csv_path = "../results/evaluation_results.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([results_row])], ignore_index=True)
    else:
        df = pd.DataFrame([results_row])
    df.to_csv(csv_path, index=False)
    print(f" Results → {csv_path}")

    return all_preds, all_labels
