"""
=============================================================================
TRAIN.PY — DistilBERT Fine-tuning for Abusive Language Detection
=============================================================================
Fine-tunes distilbert-base-uncased for 3-class sequence classification:
  Label 0 = Normal
  Label 1 = Explicit Abuse
  Label 2 = Manipulative

Uses weighted CrossEntropyLoss for class imbalance handling.
Model size: ~260MB (vs 440MB for full BERT)
=============================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from collections import Counter
import os
import time

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
SAVE_DIR = "abuse_model"


class AbuseDataset(Dataset):
    """
    PyTorch Dataset for abuse detection.
    Tokenizes text and returns input_ids, attention_mask, and labels.
    """

    def __init__(self, texts, labels, tokenizer, max_length=128):
        """
        Args:
            texts: List of text strings
            labels: List of integer labels (0, 1, or 2)
            tokenizer: BertTokenizer instance
            max_length: Maximum token length (default 128)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_class_weights(labels):
    """
    Compute inverse-frequency class weights for imbalanced datasets.
    Formula: weight_i = total_samples / (num_classes * count_i)
    """
    counter = Counter(labels)
    total = len(labels)
    n_classes = len(counter)
    weights = []
    for i in range(n_classes):
        if counter[i] > 0:
            weights.append(total / (n_classes * counter[i]))
        else:
            weights.append(1.0)
    return torch.tensor(weights, dtype=torch.float)


def train_model(train_texts, train_labels, test_texts=None, test_labels=None,
                save_dir=SAVE_DIR, num_epochs=NUM_EPOCHS):
    """
    Fine-tune BERT for 3-class abusive language detection.

    Args:
        train_texts: List of training text strings
        train_labels: List of training labels
        test_texts: Optional list of test text strings for validation
        test_labels: Optional list of test labels for validation
        save_dir: Directory to save the best model
        num_epochs: Number of training epochs

    Returns:
        model: Trained BertForSequenceClassification model
        tokenizer: BertTokenizer instance
    """
    print("=" * 60)
    print("  DistilBERT FINE-TUNING — ABUSIVE LANGUAGE DETECTION")
    print("=" * 60)

    # ---- Device Setup ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ---- Load Tokenizer ----
    print(f"\n  Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ---- Create Datasets ----
    print("  Creating PyTorch datasets...")
    train_dataset = AbuseDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_loader = None
    if test_texts is not None and test_labels is not None:
        test_dataset = AbuseDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Train batches: {len(train_loader)}")
    if test_loader:
        print(f"  Test batches:  {len(test_loader)}")

    # ---- Load Model ----
    print(f"\n  Loading model: {MODEL_NAME} (num_labels=3)")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model.to(device)

    # ---- Class Weights for Imbalanced Data ----
    class_weights = compute_class_weights(train_labels).to(device)
    print(f"  Class weights: {[round(w, 3) for w in class_weights.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ---- Optimizer + Scheduler ----
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # ---- Training Loop ----
    print(f"\n  Starting training: {num_epochs} epochs, {total_steps} total steps")
    print("-" * 60)

    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Compute weighted loss
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Track metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Progress logging every 50 batches
            if (batch_idx + 1) % 50 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"    Epoch {epoch+1}/{num_epochs} | "
                      f"Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Acc: {correct/total:.4f}")

        # Epoch summary
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        epoch_time = time.time() - start_time

        print(f"\n  === Epoch {epoch+1}/{num_epochs} Summary ===")
        print(f"    Train Loss:     {epoch_loss:.4f}")
        print(f"    Train Accuracy: {epoch_acc:.4f} ({correct}/{total})")
        print(f"    Time:           {epoch_time:.1f}s")
        print(f"    Learning Rate:  {scheduler.get_last_lr()[0]:.2e}")

        # ---- Validation ----
        if test_loader is not None:
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)

            val_acc = val_correct / val_total
            print(f"    Val Accuracy:   {val_acc:.4f} ({val_correct}/{val_total})")

            # Save best model based on validation accuracy
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                print(f"    ✅ New best model saved to {save_dir}/")

        print("-" * 60)

    # Save final model if no validation was done
    if test_loader is None:
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"\n  ✅ Model saved to {save_dir}/")

    print(f"\n  Training complete!")
    if best_accuracy > 0:
        print(f"  Best validation accuracy: {best_accuracy:.4f}")

    return model, tokenizer



