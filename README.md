# 🛡️ Abusive Language Detection in Online Conversations and Dating Platforms

> A BERT-based NLP system that detects abusive and manipulative language in online conversations using a novel **Multi-Signal Fusion Scoring System**.

---

## 📌 Project Overview

Online platforms — especially dating apps and social media — are increasingly prone to abusive and manipulative interactions. This project proposes an integrated detection framework that goes beyond simple binary classification, identifying three distinct categories of conversation:

| Label | Description | Risk Level |
|-------|-------------|------------|
| 🟢 **Normal** | Safe, healthy conversation | Low |
| 🔴 **Explicit Abuse** | Hate speech, offensive language, verbal attacks | High |
| 🟡 **Manipulative** | Gaslighting, love bombing, guilt-tripping, emotional coercion | High |

The system fine-tunes **BERT (`bert-base-uncased`)** on a multi-source dataset and combines it with a novel **Fusion Scoring System** for interpretable, real-time risk assessment.

---

## 👥 Authors

- Aaron R
- Ardra Selin
- Manasa Muraleedharan
- Abhi

**Institution:** School of Digital Sciences, Digital University Kerala (`duk.ac.in`)

---

## 🏗️ Project Structure

```
NLP PROJECT/
│
├── app.py                  # Gradio web demo interface
├── train.py                # BERT fine-tuning pipeline
├── evaluate.py             # Model evaluation & confusion matrix
├── fusion_score.py         # Multi-signal fusion scoring system
├── data_loader.py          # Dataset loading & merging pipeline
├── download_datasets.py    # Script to download DATA2 & DATA4
├── update_notebook.py      # Utility to adapt notebook for local use
├── NLP.ipynb               # Jupyter notebook (full pipeline)
├── requirements.txt        # Python dependencies
├── research_paper.md       # Full research paper (Markdown)
│
└── DATA/
    ├── DATA1/              # ConvAbuse dataset (local CSV)
    ├── DATA2/              # MentalManip majority vote (auto-downloaded)
    ├── DATA3/              # MentalManip consensus labels (local CSV)
    └── DATA4/              # ETHOS hate speech dataset (auto-downloaded)
```

---

## 📊 Datasets Used

| # | Dataset | Source | Samples | Purpose |
|---|---------|--------|---------|---------|
| DATA1 | **ConvAbuse** | Curry et al., EMNLP 2021 | ~1,400 | Conversational abuse in chatbot interactions |
| DATA2 | **MentalManip (majority)** | Wang et al., ACL 2024 | ~1,085 | Psychological manipulation in dialogues |
| DATA3 | **MentalManip (consensus)** | Wang et al., ACL 2024 | 2,915 | High-quality manipulation labels |
| DATA4 | **ETHOS** | Mollas et al., 2020 | 998 | Hate speech from YouTube & Reddit |

- Labels are merged into a unified 3-class taxonomy (Normal / Explicit Abuse / Manipulative)
- Final dataset is shuffled and split **80/20** with stratified sampling

---

## 🧠 Model Architecture

- **Base model:** `bert-base-uncased` (fine-tuned for 3-class sequence classification)
- **Loss function:** Weighted `CrossEntropyLoss` to handle class imbalance
- **Optimizer:** AdamW with linear warmup scheduler

| Hyperparameter | Value |
|----------------|-------|
| Max sequence length | 128 tokens |
| Batch size | 8–16 |
| Learning rate | 2×10⁻⁵ |
| Epochs | 3 |
| Gradient clipping | max_norm = 1.0 |

---

## ⚗️ Fusion Scoring System

A novel multi-signal risk assessment combining 3 complementary signals:

```
Final Risk Score = (0.4 × Content Score) + (0.3 × Manipulation Score) + (0.3 × Escalation Score)
```

| Signal | Weight | Method |
|--------|--------|--------|
| **Content Score** | 0.4 | BERT softmax confidence for predicted class |
| **Manipulation Score** | 0.3 | Rule-based matching of 9 high-risk phrases |
| **Escalation Score** | 0.3 | Multi-turn label escalation detection via `\|` separator |

**Risk Thresholds:**
- 🔴 `≥ 0.65` → **HIGH RISK** — Immediate action recommended
- 🟠 `≥ 0.35` → **WARNING** — Monitor and flag for review
- 🟢 `< 0.35` → **SAFE** — No action needed

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/nlp-project.git
cd nlp-project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Datasets (DATA2 & DATA4)

```bash
python download_datasets.py
```

> ⚠️ **Note:** Place the ConvAbuse dataset manually in `DATA/DATA1/2_splits/` and MentalManip consensus CSV in `DATA/DATA3/mentalmanip_dataset/`.

### 4. Train the Model

```bash
python train.py
```

The trained model will be saved to the `abuse_model/` directory.

### 5. Evaluate the Model

```bash
python evaluate.py
```

Outputs a classification report and saves a confusion matrix heatmap as `confusion_matrix.png`.

### 6. Launch the Gradio Demo

```bash
python app.py
```

Opens an interactive web UI. Use `|` to separate conversation turns for multi-turn escalation analysis.

---

## 💬 Demo Examples

| Input | Expected Label | Risk |
|-------|---------------|------|
| `"You seem really special \| Why didn't you reply? \| If you cared you wouldn't ignore me \| Send me your number now"` | Manipulative | 🔴 HIGH |
| `"You're stupid and nobody likes you"` | Explicit Abuse | 🔴 HIGH |
| `"Hey how was your day? \| Pretty good, just busy with work \| Same here!"` | Normal | 🟢 SAFE |

---

## 📦 Requirements

```
transformers
torch
datasets
pandas
scikit-learn
gradio
seaborn
matplotlib
numpy
```

---

## 📄 References

1. Curry et al. (2021). *ConvAbuse: Data, Analysis, and Benchmarks for Nuanced Abuse Detection in Conversational AI.* EMNLP 2021.
2. Wang et al. (2024). *MentalManip: A Dataset For Fine-grained Analysis of Mental Manipulation in Conversations.* ACL 2024.
3. Mollas et al. (2020). *ETHOS: an Online Hate Speech Detection Dataset.* arXiv:2006.08328.
4. Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT 2019.
5. Cécillon et al. (2019). *Abusive Language Detection in Online Conversations by Combining Content- and Graph-Based Features.* Frontiers in Big Data.

---

## 📜 License

This project is developed for academic purposes at Digital University Kerala. Please cite appropriately if you use this work.
