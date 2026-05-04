"""
=============================================================================
APP.PY — Gradio Demo Interface for Abusive Language Detection
=============================================================================
Interactive web UI that accepts conversation text and displays:
  - Predicted label (Normal / Explicit Abuse / Manipulative)
  - Content Score, Manipulation Score, Escalation Score
  - Final Risk Score and Risk Level (GREEN/ORANGE/RED)

Launch: python app.py
Share URL: Automatically generated with share=True (for Colab)
=============================================================================
"""

import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from fusion_score import compute_fusion_score, LABEL_NAMES

import os

MODEL_DIR = "abuse_model"

# ============================================================
# Load model globally (loaded once at startup)
# ============================================================
print("=" * 50)
print("  Loading Abusive Language Detection Model...")
print("=" * 50)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if os.path.exists(MODEL_DIR):
    print(f"  Loading fine-tuned model from {MODEL_DIR}...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
else:
    print(f"  ⚠️ WARNING: Fine-tuned model not found at '{MODEL_DIR}'.")
    print(f"  Falling back to base model 'bert-base-uncased' for demonstration.")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

model.to(device)
model.eval()

print(f"  Model loaded on {device}")
print(f"  Ready for predictions!")
print("=" * 50)


def analyze_conversation(text):
    """
    Main prediction function for the Gradio interface.
    Takes conversation text and returns all fusion scores.
    """
    # Handle empty input
    if not text or not text.strip():
        return "No input provided", "0.0", "0.0", "0.0", "0.0", "⚪ NO INPUT"

    # Compute fusion score
    result = compute_fusion_score(model, tokenizer, text, device)

    return (
        result['predicted_label'],
        str(result['content_score']),
        str(result['manipulation_score']),
        str(result['escalation_score']),
        str(result['final_risk_score']),
        result['risk_level']
    )


# ============================================================
# Build Gradio Interface
# ============================================================
demo = gr.Interface(
    fn=analyze_conversation,
    inputs=gr.Textbox(
        label="💬 Enter conversation (use | to separate turns)",
        placeholder="Type a conversation here...\n\n"
                    "Example: Hello, how are you? | I'm fine | "
                    "That's good to hear!",
        lines=5
    ),
    outputs=[
        gr.Textbox(label="🏷️ Predicted Label"),
        gr.Textbox(label="📊 Content Score (0.0 – 1.0)"),
        gr.Textbox(label="🎭 Manipulation Score (0.0 – 1.0)"),
        gr.Textbox(label="📈 Escalation Score (0.0 – 1.0)"),
        gr.Textbox(label="⚠️ Final Risk Score (0.0 – 1.0)"),
        gr.Textbox(label="🚦 Risk Level"),
    ],
    title="🛡️ Abusive Language Detection System",
    description="""
    **Detect abusive and manipulative language in online conversations 
    and dating platforms.**

    This system uses a fine-tuned BERT model combined with a novel 
    Fusion Scoring System to analyze conversations for three types of content:

    | Label | Description | Risk |
    |-------|-------------|------|
    | 🟢 **Normal** | Safe, healthy conversation | Low |
    | 🔴 **Explicit Abuse** | Hate speech, offensive language | High |
    | 🟡 **Manipulative** | Gaslighting, love bombing, guilt-tripping | High |

    **💡 Tip:** Use `|` to separate conversation turns for multi-turn 
    escalation analysis.
    """,
    examples=[
        ["You seem really special | Why didn't you reply? | "
         "If you cared you wouldn't ignore me | Send me your number now"],
        ["You're stupid and nobody likes you"],
        ["Hey how was your day? | Pretty good, just busy with work | "
         "Same here!"],
        ["I thought you trusted me | Everyone else understands | "
         "You owe me an explanation"],
        ["Let's grab coffee sometime | Sure that sounds fun | "
         "Great see you at 3!"],
    ],
    theme="default",
    flagging_mode="never",
)

# ============================================================
# Launch
# ============================================================
if __name__ == "__main__":
    print("\n  Launching Gradio interface...")
    print("  Use share=True for public URL (required on Colab)")
    demo.launch(share=True)
