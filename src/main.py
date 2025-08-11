

import os
from evaluate_model import evaluate_on_test
from evaluate_lstm_model import evaluate_lstm_model
from train_attention_model import train_attention_model
from train_lstm_model import train_lstm_model
from ResNet_trainer import train_resnet_model


# Configuration

SPLIT = "article_like"  # options: "article_like", "patient_dependent"
MODEL = "resnet18"      # options: "resnet18", "resnet34", "attention", "lstm"
EPOCHS = 70
BATCH_SIZE = 32

# Paths for segment CSVs

if SPLIT == "article_like":
    train_csv = "../data/generated_data/article_like_split/train/segment_level_data(128)train.csv"
    val_csv   = "../data/generated_data/article_like_split/val/segment_level_data(128)val.csv"
    test_csv  = "../data/generated_data/article_like_split/test/segment_level_data(128)test.csv"
elif SPLIT == "patient_dependent":
    train_csv = "../data/generated_data/patient-dependent/train_random.csv"
    val_csv   = "../data/generated_data/patient-dependent/val_random.csv"
    test_csv  = "../data/generated_data/patient-dependent/test_random.csv"
else:
    raise ValueError("Unknown SPLIT (use 'article_like' or 'patient_dependent')")


# Train

if MODEL in ("resnet18", "resnet34"):
    model_path = f"../models/{MODEL}_{SPLIT}_best.pth"
    train_resnet_model(
        model_type=MODEL,
        train_csv=train_csv,
        val_csv=val_csv,
        model_path=model_path,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        num_classes=3,
        in_channels=1,
    )
elif MODEL == "attention":
    model_path = f"../models/resnet34_attention_{SPLIT}.pth"
    train_attention_model(
        train_csv=train_csv,
        val_csv=val_csv,
        base_model_path=model_path,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )

else:
    raise ValueError("Unknown MODEL")

# Evaluate

if MODEL in ("resnet18", "resnet34"):
    evaluate_on_test(
        model_path=f"../models/{MODEL}_{SPLIT}_best.pth",
        test_csv=test_csv,
        model_name=f"{MODEL}_{SPLIT}_best",
        model_type=MODEL,
        batch_size=BATCH_SIZE,
        in_channels=1,
        num_classes=3,
    )
elif MODEL == "attention":
    from evaluate_attention_model import evaluate_attention_model
    evaluate_attention_model(
        model_path=f"../models/resnet34_attention_{SPLIT}.pth",
        test_csv=test_csv,
        model_name=f"resnet34_attention_{SPLIT}",
    )
