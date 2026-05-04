"""
=============================================================================
DATA LOADER - Abusive Language Detection System
=============================================================================
Loads and merges 4 datasets:
  DATA1: ConvAbuse (local CSV, explicit abuse)
  DATA2: MentalManip from HuggingFace (manipulation)
  DATA3: MentalManip from local CSV (manipulation, consensus labels)
  DATA4: ETHOS hate speech from HuggingFace

Label Mapping:
  0 = Normal (safe conversation)
  1 = Explicit Abuse (hate speech, offensive language)
  2 = Manipulative (psychological manipulation)
=============================================================================
"""

import pandas as pd
import numpy as np
import csv
import os
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION — Change BASE_DIR for Google Colab
# ============================================================
# For local use:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DATA")

# For Google Colab, uncomment below and comment above:
# BASE_DIR = "/content/drive/MyDrive/NLP PROJECT"
# DATA_DIR = os.path.join(BASE_DIR, "DATA")


def load_convabuse(data_dir):
    """
    Load ConvAbuse dataset (DATA1).
    Uses majority vote across 8 annotators to determine abuse label.
    Text is combined from prev_user and user columns.
    
    Mapping: abuse → Label 1, not abuse → Label 0
    """
    print("=" * 50)
    print("Loading DATA1: ConvAbuse Dataset")
    print("=" * 50)

    # Load train, validation, and test splits
    splits_dir = os.path.join(data_dir, "DATA1", "2_splits")
    file_names = [
        "ConvAbuseEMNLPtrain.csv",
        "ConvAbuseEMNLPvalid.csv",
        "ConvAbuseEMNLPtest.csv"
    ]

    dfs = []
    for fname in file_names:
        fpath = os.path.join(splits_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            dfs.append(df)
            print(f"  Loaded {fname}: {len(df)} rows")
        else:
            print(f"  WARNING: {fname} not found, skipping.")

    if not dfs:
        print("  ERROR: No ConvAbuse files found!")
        return pd.DataFrame(columns=['text', 'label'])

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total ConvAbuse rows: {len(df)}")

    # Process each row: combine text + majority vote for abuse label
    texts = []
    labels = []

    for idx, row in df.iterrows():
        # Combine text from prev_user and user columns
        text_parts = []
        if pd.notna(row.get('prev_user')) and str(row.get('prev_user')).strip():
            text_parts.append(str(row['prev_user']).strip())
        if pd.notna(row.get('user')) and str(row.get('user')).strip():
            text_parts.append(str(row['user']).strip())

        text = " | ".join(text_parts)
        if not text.strip():
            continue

        # Majority vote across annotators 1-8
        abuse_votes = 0
        not_abuse_votes = 0

        for ann_id in range(1, 9):
            # Abuse columns: -1 (mild), -2 (strong), -3 (very strong)
            abuse_cols = [
                f'Annotator{ann_id}_is_abuse.-1',
                f'Annotator{ann_id}_is_abuse.-2',
                f'Annotator{ann_id}_is_abuse.-3'
            ]
            not_abuse_col = f'Annotator{ann_id}_is_abuse.1'

            is_abuse = False
            for col in abuse_cols:
                if col in row.index and pd.notna(row[col]) and float(row[col]) == 1:
                    is_abuse = True
                    break

            is_not_abuse = False
            if not_abuse_col in row.index and pd.notna(row[not_abuse_col]) and float(row[not_abuse_col]) == 1:
                is_not_abuse = True

            if is_abuse:
                abuse_votes += 1
            elif is_not_abuse:
                not_abuse_votes += 1

        total_votes = abuse_votes + not_abuse_votes
        if total_votes == 0:
            continue  # No annotator labeled this row

        # Majority rule: >= 50% abuse votes → Label 1
        if abuse_votes / total_votes >= 0.5:
            labels.append(1)  # Explicit Abuse
        else:
            labels.append(0)  # Normal

        texts.append(text)

    result = pd.DataFrame({'text': texts, 'label': labels})
    print(f"  ConvAbuse processed: {len(result)} rows")
    print(f"  Label distribution: {result['label'].value_counts().to_dict()}")
    return result


def load_mentalmanip_local(data_dir):
    """
    Load MentalManip from local CSV (DATA3).
    Uses csv module as recommended by dataset authors (pandas has parsing issues).
    File: mentalmanip_con.csv (consensus labels, 2915 dialogues)
    
    Mapping: Manipulative=1 → Label 2, Manipulative=0 → Label 0
    """
    print("\n" + "=" * 50)
    print("Loading DATA3: MentalManip (Local CSV)")
    print("=" * 50)

    csv_path = os.path.join(data_dir, "DATA3", "mentalmanip_dataset", "mentalmanip_con.csv")

    if not os.path.exists(csv_path):
        print(f"  ERROR: {csv_path} not found!")
        return pd.DataFrame(columns=['text', 'label', 'source_id'])

    texts = []
    labels = []
    ids = []

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        header = next(reader)

        # Find column indices
        id_idx = header.index('ID')
        dialogue_idx = header.index('Dialogue')
        manip_idx = header.index('Manipulative')

        for row in reader:
            if len(row) <= max(id_idx, dialogue_idx, manip_idx):
                continue

            text = row[dialogue_idx].strip()
            manip = row[manip_idx].strip()

            if not text or not manip:
                continue

            try:
                manip_val = int(manip)
            except ValueError:
                continue

            texts.append(text)
            ids.append(row[id_idx])

            if manip_val == 1:
                labels.append(2)  # Manipulative
            else:
                labels.append(0)  # Normal

    result = pd.DataFrame({'text': texts, 'label': labels, 'source_id': ids})
    print(f"  MentalManip local: {len(result)} rows")
    print(f"  Label distribution: {result['label'].value_counts().to_dict()}")
    return result


def load_mentalmanip_hf(data_dir, local_ids=None):
    """
    Load MentalManip from HuggingFace (DATA2).
    Uses mentalmanip_maj config (4000 dialogues, majority vote labels).
    Deduplicates against local DATA3 to avoid overlap.
    
    Mapping: Manipulative=1 → Label 2, Manipulative=0 → Label 0
    """
    print("\n" + "=" * 50)
    print("Loading DATA2: MentalManip (HuggingFace)")
    print("=" * 50)

    # Check if already downloaded
    saved_path = os.path.join(data_dir, "DATA2", "mentalmanip_hf.csv")
    if os.path.exists(saved_path):
        print(f"  Found cached file: {saved_path}")
        result = pd.read_csv(saved_path)
        print(f"  Loaded {len(result)} rows from cache")
        print(f"  Label distribution: {result['label'].value_counts().to_dict()}")
        return result

    try:
        from datasets import load_dataset

        print("  Downloading from HuggingFace...")
        dataset = load_dataset("audreyeleven/MentalManip", "mentalmanip_maj")

        texts = []
        labels = []

        for split in dataset:
            for item in dataset[split]:
                dialogue = item.get('Dialogue', '')
                manip = item.get('Manipulative', None)
                item_id = str(item.get('ID', ''))

                if not dialogue or manip is None:
                    continue

                # Skip if already in local dataset (deduplication)
                if local_ids and item_id in local_ids:
                    continue

                texts.append(dialogue.strip())
                if int(manip) == 1:
                    labels.append(2)  # Manipulative
                else:
                    labels.append(0)  # Normal

        result = pd.DataFrame({'text': texts, 'label': labels})

        # Save to DATA2 folder for future use
        save_dir = os.path.join(data_dir, "DATA2")
        os.makedirs(save_dir, exist_ok=True)
        result.to_csv(saved_path, index=False)

        print(f"  MentalManip HF: {len(result)} rows (after dedup)")
        print(f"  Label distribution: {result['label'].value_counts().to_dict()}")
        print(f"  Saved to {saved_path}")
        return result

    except Exception as e:
        print(f"  WARNING: Could not load from HuggingFace: {e}")
        print("  Skipping DATA2.")
        return pd.DataFrame(columns=['text', 'label'])


def load_ethos(data_dir):
    """
    Load ETHOS hate speech dataset (DATA4) from HuggingFace.
    Binary version: 998 comments (433 hate + 565 non-hate).
    
    Mapping: label=1 (hate) → Label 1, label=0 → Label 0
    """
    print("\n" + "=" * 50)
    print("Loading DATA4: ETHOS Dataset")
    print("=" * 50)

    # Check if already downloaded
    saved_path = os.path.join(data_dir, "DATA4", "ethos_binary.csv")
    if os.path.exists(saved_path):
        print(f"  Found cached file: {saved_path}")
        result = pd.read_csv(saved_path)
        print(f"  Loaded {len(result)} rows from cache")
        print(f"  Label distribution: {result['label'].value_counts().to_dict()}")
        return result

    try:
        from datasets import load_dataset

        print("  Downloading from HuggingFace...")
        dataset = load_dataset("iamollas/ethos", "binary", trust_remote_code=True)

        texts = []
        labels = []

        for split in dataset:
            for item in dataset[split]:
                text = item.get('text', '')
                label = item.get('label', None)

                if not text or label is None:
                    continue

                texts.append(str(text).strip())
                if int(label) == 1:
                    labels.append(1)  # Hate speech → Explicit Abuse
                else:
                    labels.append(0)  # Normal

        result = pd.DataFrame({'text': texts, 'label': labels})

        # Save to DATA4 folder
        save_dir = os.path.join(data_dir, "DATA4")
        os.makedirs(save_dir, exist_ok=True)
        result.to_csv(saved_path, index=False)

        print(f"  ETHOS: {len(result)} rows")
        print(f"  Label distribution: {result['label'].value_counts().to_dict()}")
        print(f"  Saved to {saved_path}")
        return result

    except Exception as e:
        print(f"  WARNING: Could not load ETHOS: {e}")
        print("  Skipping DATA4.")
        return pd.DataFrame(columns=['text', 'label'])


def load_and_merge_all(data_dir=None):
    """
    Master function: Load all 4 datasets, merge, clean, shuffle, and split.
    Returns: train_texts, test_texts, train_labels, test_labels, full_df
    """
    if data_dir is None:
        data_dir = DATA_DIR

    print("\n" + "=" * 60)
    print("  ABUSIVE LANGUAGE DETECTION — DATA PIPELINE")
    print("=" * 60)

    # Step 1: Load ConvAbuse (DATA1) — Explicit abuse
    df_convabuse = load_convabuse(data_dir)

    # Step 2: Load MentalManip local (DATA3) — get IDs for dedup
    df_mm_local = load_mentalmanip_local(data_dir)
    local_ids = set(df_mm_local['source_id'].astype(str).values) if 'source_id' in df_mm_local.columns else set()

    # Step 3: Load MentalManip HF (DATA2) — deduplicated
    df_mm_hf = load_mentalmanip_hf(data_dir, local_ids)

    # Step 4: Load ETHOS (DATA4) — Hate speech
    df_ethos = load_ethos(data_dir)

    # Drop helper columns before merge
    if 'source_id' in df_mm_local.columns:
        df_mm_local = df_mm_local.drop(columns=['source_id'])

    # Step 5: Merge all datasets
    print("\n" + "=" * 50)
    print("MERGING ALL DATASETS")
    print("=" * 50)

    all_dfs = [df_convabuse, df_mm_local, df_mm_hf, df_ethos]
    df_merged = pd.concat(all_dfs, ignore_index=True)

    # Step 6: Clean
    df_merged = df_merged.dropna(subset=['text', 'label'])
    df_merged = df_merged[df_merged['text'].str.strip().str.len() > 0]
    df_merged['label'] = df_merged['label'].astype(int)
    df_merged = df_merged.reset_index(drop=True)

    # Step 7: Shuffle
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n  Total merged dataset: {len(df_merged)} rows")
    print(f"  Label distribution:")
    label_names = {0: "Normal", 1: "Explicit Abuse", 2: "Manipulative"}
    for label_id in sorted(df_merged['label'].unique()):
        count = (df_merged['label'] == label_id).sum()
        print(f"    Label {label_id} ({label_names.get(label_id, '?')}): {count}")

    # Step 8: Train/Test split (80/20, stratified)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df_merged['text'].tolist(),
        df_merged['label'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df_merged['label'].tolist()
    )

    print(f"\n  Train set: {len(train_texts)} samples")
    print(f"  Test set:  {len(test_texts)} samples")
    print("  ✅ Data pipeline complete!")

    return train_texts, test_texts, train_labels, test_labels, df_merged



