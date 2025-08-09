# dataset_preparer.py

import os
import pandas as pd
import glob

def extract_murmur_label(subject_file):
    label = "Unknown"
    with open(subject_file, "r") as file:
        for line in file:
            if line.startswith("#Murmur:"):
                label = line.split(":")[1].strip()
    return label

def load_patient_dataset(dataset_path="../data", output_path="../data/generated_data/processed_data.csv"):
    """
    Loads metadata, murmur labels, and .wav file paths into a single DataFrame.
    Returns: pandas DataFrame
    """
    audio_dir = os.path.join(dataset_path, "training_data")
    csv_path = os.path.join(dataset_path, "training_data.csv")

    # Load training_data.csv
    df = pd.read_csv(csv_path)
    df.rename(columns={"Patient ID": "subject_id"}, inplace=True)
    df["subject_id"] = df["subject_id"].astype(str)

    # Load murmur labels from .txt files
    subject_files = glob.glob(os.path.join(audio_dir, "*.txt"))
    label_data = []
    for file in subject_files:
        subject_id = os.path.basename(file).split(".")[0]
        murmur_label = extract_murmur_label(file)
        label_data.append({"subject_id": subject_id, "murmur_label": murmur_label})
    label_df = pd.DataFrame(label_data)
    label_df["subject_id"] = label_df["subject_id"].astype(str)

    # Merge murmur labels into main data
    df = df.merge(label_df, on="subject_id", how="left")

    # Gather all .wav file paths
    wav_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    audio_data = []
    for file in wav_files:
        filename = os.path.basename(file)
        subject_id = filename.split("_")[0]
        location_code = filename.split("_")[1].split(".")[0]
        audio_data.append({
            "subject_id": subject_id,
            "file_path": file,
            "auscultation_location": location_code
        })

    audio_df = pd.DataFrame(audio_data)
    audio_df["subject_id"] = audio_df["subject_id"].astype(str)

    # Merge audio with full metadata
    full_df = audio_df.merge(df, on="subject_id", how="left")

    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    full_df.to_csv(output_path, index=False)

    print("✅ Patient dataset loaded and saved!")
    return full_df
