"""
=============================================================================
FUSION_SCORE.PY — Multi-Signal Fusion Scoring System
=============================================================================
Novel risk assessment combining 3 signals:
  1. Content Score:      BERT softmax confidence for predicted class
  2. Manipulation Score: Rule-based keyword matching (9 high-risk phrases)
  3. Escalation Score:   Multi-turn risk escalation detection

Final Risk = (0.4 × Content) + (0.3 × Manipulation) + (0.3 × Escalation)

Risk Levels:
  >= 0.65 → 🔴 HIGH RISK
  >= 0.35 → 🟠 WARNING
  <  0.35 → 🟢 SAFE
=============================================================================
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABEL_NAMES = ["Normal", "Explicit Abuse", "Manipulative"]
MODEL_DIR = "abuse_model"

# ============================================================
# High-risk manipulation phrases (rule-based detector)
# ============================================================
MANIPULATION_PHRASES = [
    "send me your number",
    "if you cared",
    "you never",
    "prove it",
    "don't tell anyone",
    "only you understand me",
    "everyone else",
    "you owe me",
    "i thought you trusted me"
]


def compute_content_score(model, tokenizer, text, device):
    """
    Compute content score using BERT softmax probability.
    Returns the predicted label, confidence score, and all class probabilities.
    """
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(probs, dim=-1).item()
        content_score = probs[0][pred_label].item()

    return pred_label, content_score, probs[0].cpu().numpy()


def compute_manipulation_score(text):
    """
    Rule-based manipulation keyword checker.
    Scans text for 9 high-risk manipulation phrases.
    Returns: score (0.0 to 1.0) and number of matches found.
    """
    text_lower = text.lower()
    matches = sum(1 for phrase in MANIPULATION_PHRASES if phrase in text_lower)
    score = matches / len(MANIPULATION_PHRASES)
    return score, matches


def compute_escalation_score(model, tokenizer, text, device):
    """
    Multi-turn escalation detection.
    Splits input by '|' separator, runs BERT on each turn,
    and checks if predicted labels escalate (increase) over turns.
    
    Returns 0.8 if escalation detected, 0.2 otherwise.
    """
    turns = [t.strip() for t in text.split("|") if t.strip()]

    # Single turn → no escalation possible
    if len(turns) <= 1:
        return 0.2

    # Get BERT prediction for each turn
    turn_labels = []
    for turn in turns:
        label, _, _ = compute_content_score(model, tokenizer, turn, device)
        turn_labels.append(label)

    # Check for escalation: does the label increase over turns?
    escalation = False
    for i in range(1, len(turn_labels)):
        if turn_labels[i] > turn_labels[i - 1]:
            escalation = True
            break

    return 0.8 if escalation else 0.2


def compute_fusion_score(model, tokenizer, text, device):
    """
    Compute the complete fusion risk score combining all 3 signals.

    Formula: Final = (0.4 × Content) + (0.3 × Manipulation) + (0.3 × Escalation)

    Args:
        model: Trained BertForSequenceClassification
        tokenizer: BertTokenizer
        text: Input conversation text (use | for multi-turn)
        device: torch device

    Returns:
        Dictionary with all scores and risk assessment.
    """
    # Signal 1: Content score (BERT confidence)
    pred_label, content_score, all_probs = compute_content_score(
        model, tokenizer, text, device
    )

    # Signal 2: Manipulation score (rule-based keywords)
    manipulation_score, num_matches = compute_manipulation_score(text)

    # Signal 3: Escalation score (multi-turn analysis)
    escalation_score = compute_escalation_score(model, tokenizer, text, device)

    # Fusion formula
    final_score = (0.4 * content_score) + (0.3 * manipulation_score) + (0.3 * escalation_score)

    # Risk level thresholding
    if final_score >= 0.65:
        risk_level = "🔴 HIGH RISK"
    elif final_score >= 0.35:
        risk_level = "🟠 WARNING"
    else:
        risk_level = "🟢 SAFE"

    result = {
        'predicted_label': LABEL_NAMES[pred_label],
        'predicted_label_id': pred_label,
        'content_score': round(content_score, 4),
        'manipulation_score': round(manipulation_score, 4),
        'manipulation_matches': num_matches,
        'escalation_score': round(escalation_score, 4),
        'final_risk_score': round(final_score, 4),
        'risk_level': risk_level,
        'class_probabilities': {
            LABEL_NAMES[i]: round(float(all_probs[i]), 4) for i in range(3)
        }
    }

    return result


def load_model_and_score(text, model_dir=MODEL_DIR):
    """Convenience function: load model and compute fusion score."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return compute_fusion_score(model, tokenizer, text, device)


def print_result(result, text=""):
    """Pretty-print a fusion score result."""
    if text:
        display = text[:80] + "..." if len(text) > 80 else text
        print(f"  Input: \"{display}\"")
    print(f"  Predicted Label:    {result['predicted_label']}")
    print(f"  Content Score:      {result['content_score']}")
    print(f"  Manipulation Score: {result['manipulation_score']} "
          f"({result['manipulation_matches']} phrase matches)")
    print(f"  Escalation Score:   {result['escalation_score']}")
    print(f"  Final Risk Score:   {result['final_risk_score']}")
    print(f"  Risk Level:         {result['risk_level']}")
    print(f"  Class Probabilities: {result['class_probabilities']}")



