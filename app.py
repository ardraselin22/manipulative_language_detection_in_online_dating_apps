"""
=============================================================================
APP.PY — Streamlit Web Interface for Abusive Language Detection
=============================================================================
Interactive web UI that accepts conversation text and displays:
  - Predicted label (Normal / Explicit Abuse / Manipulative)
  - Content Score, Manipulation Score, Escalation Score
  - Final Risk Score and Risk Level (GREEN/ORANGE/RED)

Deploy: Streamlit Cloud (auto-detects app.py)
=============================================================================
"""

import streamlit as st
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fusion_score import compute_fusion_score, LABEL_NAMES

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Abusive Language Detection",
    page_icon="🛡️",
    layout="centered"
)

# ============================================================
# Load Model (cached so it only loads once)
# ============================================================
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = "abuse_model"

    if os.path.exists(model_dir):
        # Check if model.safetensors is a Git LFS pointer (typically very small, < 1KB)
        model_file = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(model_file) and os.path.getsize(model_file) < 1024:
            import requests
            st.warning("🔄 **First Boot:** Downloading model weights from Git LFS (~255MB)... This will take a moment.")
            url = "https://media.githubusercontent.com/media/ardraselin22/manipulative_language_detection_in_online_dating_apps/main/abuse_model/model.safetensors"
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(model_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("✅ Download complete!")
            
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=3
        )

    model.to(device)
    model.eval()
    return model, tokenizer, device


model, tokenizer, device = load_model()

# ============================================================
# UI Layout
# ============================================================
st.title("🛡️ Abusive Language Detection System")
st.markdown(
    """
    **Detect abusive and manipulative language in online conversations 
    and dating platforms.**

    This system uses a fine-tuned BERT model combined with a novel 
    **Fusion Scoring System** to analyze conversations for three types of content:

    | Label | Description | Risk |
    |-------|-------------|------|
    | 🟢 **Normal** | Safe, healthy conversation | Low |
    | 🔴 **Explicit Abuse** | Hate speech, offensive language | High |
    | 🟡 **Manipulative** | Gaslighting, love bombing, guilt-tripping | High |

    **💡 Tip:** Use `|` to separate conversation turns for multi-turn 
    escalation analysis.
    """
)

st.divider()

# ============================================================
# Input
# ============================================================
if "input_text" not in st.session_state:
    st.session_state["input_text"] = ""

def set_example_text(text_val):
    st.session_state["input_text"] = text_val

text = st.text_area(
    "💬 Enter conversation (use | to separate turns)",
    placeholder="Type a conversation here...\n\n"
                "Example: Hello, how are you? | I'm fine | "
                "That's good to hear!",
    height=120,
    key="input_text"
)

# Example buttons
st.markdown("**Try an example:**")
col1, col2, col3 = st.columns(3)
with col1:
    st.button("🟡 Manipulative", on_click=set_example_text, args=("You seem really special | Why didn't you reply? | If you cared you wouldn't ignore me | Send me your number now",), use_container_width=True)
with col2:
    st.button("🔴 Abusive", on_click=set_example_text, args=("You're stupid and nobody likes you",), use_container_width=True)
with col3:
    st.button("🟢 Normal", on_click=set_example_text, args=("Hey how was your day? | Pretty good, just busy with work | Same here!",), use_container_width=True)

st.divider()

# ============================================================
# Prediction
# ============================================================
predict_button = st.button("🔍 Predict", type="primary", use_container_width=True)

if predict_button:
    if text and text.strip():
        with st.spinner("Analyzing conversation..."):
            result = compute_fusion_score(model, tokenizer, text, device)

        # Risk level color
        risk = result['risk_level']
        if "HIGH" in risk:
            st.error(f"**{risk}**")
        elif "WARNING" in risk:
            st.warning(f"**{risk}**")
        else:
            st.success(f"**{risk}**")

        # Predicted label
        st.subheader(f"🏷️ Predicted: {result['predicted_label']}")

        # Score cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📊 Content", f"{result['content_score']:.4f}")
        c2.metric("🎭 Manipulation", f"{result['manipulation_score']:.4f}")
        c3.metric("📈 Escalation", f"{result['escalation_score']:.4f}")
        c4.metric("⚠️ Final Risk", f"{result['final_risk_score']:.4f}")

        # Details
        with st.expander("📋 Full Details"):
            st.json({
                "predicted_label": result['predicted_label'],
                "content_score": result['content_score'],
                "manipulation_score": result['manipulation_score'],
                "manipulation_matches": result['manipulation_matches'],
                "escalation_score": result['escalation_score'],
                "final_risk_score": result['final_risk_score'],
                "risk_level": result['risk_level'],
                "class_probabilities": result['class_probabilities'],
            })
    else:
        st.warning("⚠️ Please enter a conversation above to analyze it.")
else:
    st.info("👆 Enter a conversation above and click **Predict** to analyze it.")

# ============================================================
# Footer
# ============================================================
st.divider()
st.caption(
    "Built with BERT + Fusion Scoring | "
    "NLP Project — Manipulative Language Detection in Online Dating Apps"
)
