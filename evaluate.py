"""
=============================================================================
EVALUATE.PY — Model Evaluation for Abusive Language Detection
=============================================================================
Generates:
  - Classification report (precision, recall, F1 per class)
  - Confusion matrix heatmap (saved as PNG)
  - Overall accuracy score
=============================================================================
"""

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
import os

# Label names for the 3 classes
LABEL_NAMES = ["Normal", "Explicit Abuse", "Manipulative"]
MODEL_DIR = "abuse_model"


def load_model(model_dir=MODEL_DIR):
    """Load the saved BERT model and tokenizer."""
    print(f"  Loading model from: {model_dir}")
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print(f"  Model loaded on {device}")
    return model, tokenizer, device


def get_predictions(model, tokenizer, texts, device, batch_size=16, max_length=128):
    """
    Run model predictions on a list of texts.
    Returns predicted labels and probability arrays.
    """
    from train import AbuseDataset

    # Create dataset with dummy labels
    dummy_labels = [0] * len(texts)
    dataset = AbuseDataset(texts, dummy_labels, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_probs = []

    print(f"  Running predictions on {len(texts)} samples...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if (batch_idx + 1) % 50 == 0:
                print(f"    Processed {(batch_idx+1) * batch_size}/{len(texts)} samples")

    return np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(true_labels, pred_labels, save_path="confusion_matrix.png"):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title('Confusion Matrix — Abusive Language Detection',
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✅ Confusion matrix saved to {save_path}")
    return cm


def evaluate_model(test_texts, test_labels, model_dir=MODEL_DIR):
    """
    Complete evaluation pipeline.
    Prints classification report, saves confusion matrix, returns metrics.
    """
    print("=" * 60)
    print("  MODEL EVALUATION")
    print("=" * 60)

    # Load model
    model, tokenizer, device = load_model(model_dir)

    # Get predictions
    predictions, probabilities = get_predictions(model, tokenizer, test_texts, device)

    # ---- Overall Accuracy ----
    accuracy = accuracy_score(test_labels, predictions)
    print(f"\n  {'='*50}")
    print(f"  OVERALL ACCURACY: {accuracy:.4f} ({int(accuracy * len(test_labels))}/{len(test_labels)})")
    print(f"  {'='*50}")

    # ---- Classification Report ----
    print(f"\n  CLASSIFICATION REPORT")
    print(f"  {'-'*50}")
    report = classification_report(
        test_labels, predictions,
        target_names=LABEL_NAMES,
        digits=4
    )
    print(report)

    # ---- Confusion Matrix ----
    print("  Generating confusion matrix...")
    cm = plot_confusion_matrix(test_labels, predictions)

    # ---- Per-class Analysis ----
    print(f"\n  PER-CLASS ANALYSIS")
    print(f"  {'-'*50}")
    for i, name in enumerate(LABEL_NAMES):
        true_count = sum(1 for l in test_labels if l == i)
        pred_count = sum(1 for p in predictions if p == i)
        correct = sum(1 for t, p in zip(test_labels, predictions) if t == i and p == i)
        print(f"    {name}: True={true_count}, Predicted={pred_count}, Correct={correct}")

    return accuracy, report, cm



