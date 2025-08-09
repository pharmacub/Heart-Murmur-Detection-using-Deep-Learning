import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_segmented_data_by_subject(input_csv="../data/generated_data/multi_windows/128_mels/segment_level_data(128)train.csv",
                                     output_dir="../data/generated_data/multi_windows/128_mels/",
                                     subject_column="subject_id",
                                     label_column="murmur_label",
                                     test_size=0.15,
                                     val_size=0.15,
                                     random_state=42):
    df = pd.read_csv(input_csv)
    subject_df = df[[subject_column, label_column]].drop_duplicates()

    train_subjects, temp_subjects = train_test_split(
        subject_df,
        test_size=val_size + test_size,
        stratify=subject_df[label_column],
        random_state=random_state
    )

    val_ratio = val_size / (val_size + test_size)
    val_subjects, test_subjects = train_test_split(
        temp_subjects,
        test_size=1 - val_ratio,
        stratify=temp_subjects[label_column],
        random_state=random_state
    )

    train_df = df[df[subject_column].isin(train_subjects[subject_column])]
    val_df = df[df[subject_column].isin(val_subjects[subject_column])]
    test_df = df[df[subject_column].isin(test_subjects[subject_column])]

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train_segments.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_segments.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_segments.csv"), index=False)

    # Print formatted table
    def summarize(name, data):
        total = len(data)
        summary = data[label_column].value_counts()
        summary_ratio = data[label_column].value_counts(normalize=True)
        return {
            "Split": name,
            "Total": total,
            **{f"{label}": f"{summary[label]} ({summary_ratio[label]:.2%})"
               for label in summary.index}
        }

    print("\n✅ Dataset split complete and saved:")
    summaries = [summarize("Train", train_df),
                 summarize("Validation", val_df),
                 summarize("Test", test_df)]

    # Convert to table
    summary_df = pd.DataFrame(summaries)
    print("\n📊 Final Segment-Level Split Summary:")
    print(summary_df.to_string(index=False))

    return train_df, val_df, test_df


def split_segmented_data_patient_dependent(
    input_csv="../data/generated_data/patient-dependent/segment_level_data(128).csv",
    output_dir="../data/generated_data/patient-dependent/",
    label_column="murmur_label",
    test_size=0.30,
    val_size=0.10,
    random_state=42):

    df = pd.read_csv(input_csv)

    # First split off test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_column],
        random_state=random_state
    )

    # Then split train+val
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        stratify=train_val_df[label_column],
        random_state=random_state
    )

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train_random.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_random.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_random.csv"), index=False)

    # Summary
    def summarize(name, data):
        total = len(data)
        summary = data[label_column].value_counts()
        ratio = data[label_column].value_counts(normalize=True)
        return {
            "Split": name,
            "Total": total,
            **{f"{k}": f"{summary[k]} ({ratio[k]:.2%})" for k in summary.index}
        }

    print("\n✅ Patient-dependent random split completed:")
    summary_df = pd.DataFrame([
        summarize("Train", train_df),
        summarize("Validation", val_df),
        summarize("Test", test_df)
    ])
    print("\n📊 Segment-Level Split Summary:")
    print(summary_df.to_string(index=False))

    return train_df, val_df, test_df


def split_patient_location_dependent(
    input_csv="../data/generated_data/patient-dependent/segment_level_data(128).csv",
    output_dir="../data/generated_data/location-dependent/",
    segment_column="segment_path",  # column containing "50336_MV_seg0.npy"
    label_column="murmur_label",
    test_size=0.30,
    val_size=0.10,
    random_state=42):

    df = pd.read_csv(input_csv)

    # Extract patient_location: e.g., "50336_MV" from "50336_MV_seg0.npy"
    df["patient_location"] = df[segment_column].apply(lambda x: os.path.basename(x).split("_seg")[0])

    unique_groups = df["patient_location"].unique()

    # Split patient-location groups
    train_val_groups, test_groups = train_test_split(
        unique_groups, test_size=test_size, random_state=random_state
    )
    val_ratio = val_size / (1 - test_size)
    train_groups, val_groups = train_test_split(
        train_val_groups, test_size=val_ratio, random_state=random_state
    )

    # Assign splits based on group membership
    train_df = df[df["patient_location"].isin(train_groups)]
    val_df = df[df["patient_location"].isin(val_groups)]
    test_df = df[df["patient_location"].isin(test_groups)]

    # Save to files
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train_patient_location.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_patient_location.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_patient_location.csv"), index=False)

    # Summary
    def summarize(name, data):
        total = len(data)
        counts = data[label_column].value_counts()
        ratios = data[label_column].value_counts(normalize=True)
        return {
            "Split": name,
            "Total": total,
            **{label: f"{counts[label]} ({ratios[label]:.2%})" for label in counts.index}
        }

    print("\n✅ Patient-location dependent random split completed:")
    print(pd.DataFrame([summarize("Train", train_df),
                        summarize("Validation", val_df),
                        summarize("Test", test_df)]).to_string(index=False))

    return train_df, val_df, test_df

import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_segmented_data_like_article(input_csv="../data/generated_data/multi_windows/128_mels/spectrogram_data(128).csv",
                                      output_dir="../data/generated_data/article_like_split/",
                                      label_column="murmur_label",
                                      random_state=42):
    # === Load dataset ===
    df = pd.read_csv(input_csv)

    # === Define exact split ratios as per the reference article ===
    test_ratio = 0.308  # ~30.8%
    val_ratio_within_remaining = 0.092 / (1 - test_ratio)  # Adjusted to ~9.2% overall

    # === First split: test vs. (train+val) ===
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df[label_column],
        random_state=random_state
    )

    # === Second split: train vs. val ===
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_within_remaining,
        stratify=train_val_df[label_column],
        random_state=random_state
    )

    # === Save the splits ===
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train_article_split.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_article_split.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_article_split.csv"), index=False)

    # === Summary Printout ===
    def summarize(name, data):
        total = len(data)
        summary = data[label_column].value_counts()
        summary_ratio = data[label_column].value_counts(normalize=True)
        return {
            "Split": name,
            "Total": total,
            **{f"{label}": f"{summary[label]} ({summary_ratio[label]:.2%})"
               for label in summary.index}
        }

    summaries = [
        summarize("Train", train_df),
        summarize("Validation", val_df),
        summarize("Test", test_df)
    ]

    summary_df = pd.DataFrame(summaries)
    print("\n📊 Final Article-like Segment-Level Split Summary:")
    print(summary_df.to_string(index=False))

    return train_df, val_df, test_df


def split_segmented_data_randomly():
    return None