# ShieldAI - Phishing Email Detector
**CYBERSEC 520 Final Project · Duke University · April 2026**

Name: Eric Ortega Rodriguez

A side-by-side comparison of traditional ML and LLM-based phishing detection, deployed as an interactive Streamlit application. Built for CYBERSEC 520 to explore how traditional ML and LLMs handle evolving phishing threats.

---

## Live Demo

> [ShieldAI on Streamlit Cloud](https://cybersec520-phishing-detector-qeylmuvykpjrvp96rarklg.streamlit.app/)

## Demo Video

> [ShieldAI Demo on YouTube](https://youtu.be/XDcSAY4FOeA)

---

## Project Overview

This project builds and compares two approaches to phishing email detection:

- **ML Models (Random Forest + XGBoost)** - trained on the CEAS_08 dataset using TF-IDF vectorization and 11 handcrafted signal features
- **LLM Agent (GPT 4.1 Mini via Duke LiteLLM)** - a tool-calling agent that reasons semantically about email intent using 4 investigative tools

Key finding: both ML models achieve **99% accuracy on the primary dataset** but collapse to **~60% on SpamAssassin** (secondary dataset) - a 39-point generalization gap that demonstrates why LLM-based detection is more robust to evolving phishing attacks.

---

## Repository Structure

```
cybersec520-phishing-detector/
├── data/
│   ├── CEAS_08.csv                    # Primary training dataset (39,154 emails)
│   ├── spam_assassin.csv              # Secondary generalizability dataset (5,796 emails)
│   └── demo_samples.csv               # Curated demo emails for the app
├── models/
│   ├── xgb_model.pkl                  # Trained XGBoost model
│   ├── tfidf_vectorizer.pkl           # Fitted TF-IDF vectorizer
│   ├── confusion_matrices.png
│   ├── confusion_matrices_spamassassin.png
│   ├── roc_curves.png
│   └── model_comparison.png
├── notebook/
│   └── phishing_detector.ipynb        # Full analysis, training, and evaluation
├── app.py                             # Streamlit app (ShieldAI)
├── requirements.txt
├── env.example                        # Environment variable template (no secrets)
├── ShieldAI_Presentation.pdf          # Final presentation slides
└── README.md
```

---

## Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/ericiortega/cybersec520-phishing-detector
cd cybersec520-phishing-detector
```

**2. Create and activate virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set environment variables**

Copy `env.example` to `.env` and fill in your credentials:
```bash
cp env.example .env
```

Edit `.env`:
```
LITELLM_TOKEN=your-duke-litellm-key
LITELLM_URL=https://litellm.oit.duke.edu
```

**5. Run the app**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Datasets

| Dataset | Size | Role |
|---|---|---|
| CEAS_08 | 39,154 emails | Primary training & evaluation dataset |
| SpamAssassin | 5,796 emails | Secondary generalizability test only (never used in training) |

> [CEAS_08 on Kaggle](https://www.kaggle.com/datasets/doryanay/ceas-08?resource=download) · [SpamAssassin on Kaggle](https://www.kaggle.com/datasets/ganiyuolalekan/spam-assassin-email-classification-dataset)

---

## Model Results

| Dataset | Model | Accuracy | F1 (weighted) | AUC-ROC |
|---|---|---|---|---|
| CEAS_08 (Primary) | Random Forest | 98% | 98% | 0.9993 |
| CEAS_08 (Primary) | XGBoost | **99%** | **99%** | **0.9994** |
| SpamAssassin (Secondary) | Random Forest | 63% | 60% | 0.5383 |
| SpamAssassin (Secondary) | XGBoost | 60% | 57% | 0.5442 |

---

### Key Finding: The 39-Point Generalization Gap

Both models collapse from ~99% to ~60% accuracy on SpamAssassin — a 39-point drop caused by distribution shift. Critically, **both models achieve near-zero phishing recall on SpamAssassin**, defaulting to predicting the majority of emails as legitimate. Three root causes:

1. **Vocabulary mismatch** - TF-IDF vocabulary was learned from 2008-era CEAS patterns; SpamAssassin uses different terminology
2. **Class distribution shift** - CEAS_08 is 56% phishing vs SpamAssassin's 33%, miscalibrating the decision boundary
3. **Feature distribution shift** - handcrafted features (urgency keywords, free email domains) are less predictive in SpamAssassin

This is precisely why LLM-based detection is more robust: the LLM agent reasons about **intent and context**, not surface patterns — making it adaptive to evolving attacks without retraining.

---

## LLM Agent - How It Works

The agent uses a 4-tool investigation loop via the OpenAI tool-calling API:

| Step | Tool | What It Checks |
|---|---|---|
| 1 | `analyze_sender` | Domain spoofing, free email providers, brand impersonation |
| 2 | `check_urgency` | Social engineering tactics - urgency, fear, greed language |
| 3 | `extract_urls` | Suspicious URLs, raw IPs, URL shorteners, obfuscated links |
| 4 | `assess_context` | Credential requests, unusual payment methods, suspicious intent |

After all tools run, the agent produces:
- **VERDICT**: PHISHING or LEGITIMATE
- **CONFIDENCE**: HIGH / MEDIUM / LOW
- **KEY RED FLAGS**: specific findings per tool
- **EXECUTIVE SUMMARY**: plain English for a non-technical audience

---

## Deployment

The app is deployed on Streamlit Cloud.

**Required secrets** (set in Streamlit Cloud → App Settings → Secrets):

| Secret | Value |
|---|---|
| `LITELLM_TOKEN` | Duke AI Gateway API key |
| `LITELLM_URL` | `https://litellm.oit.duke.edu` |

**Deployment steps:**
1. Push repo to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repo → select `app.py` as main file
4. Add secrets under Advanced Settings
5. Click Deploy

---

## Requirements

```
streamlit
pandas
numpy
scikit-learn
xgboost
scipy
openai
python-dotenv
plotly
```

Install all with:
```bash
pip install -r requirements.txt
```

> **macOS users:** XGBoost requires `libomp`. Install with `brew install libomp` if you get an import error.

---

## Notebook

The Jupyter notebook (`notebook/phishing_detector.ipynb`) covers:

- **Section 0** - Threat model (threat actor, targets, evasion, operational context)
- **Section 1** - Setup and imports
- **Section 2** - Data loading and exploratory analysis
- **Section 3** - Preprocessing and feature engineering (TF-IDF + 11 handcrafted features)
- **Section 4** - Model training (Random Forest + XGBoost)
- **Section 5** - Primary dataset evaluation (CEAS_08)
- **Section 6** - Generalizability testing (SpamAssassin) including threshold tuning analysis
- **Section 7** - Model export and demo sample generation

---

## GenAI Usage

*Claude (Anthropic) assisted with code formatting, Streamlit UI development, and regex parsing of the SpamAssassin dataset; all core logic, modeling, and analysis are my own.*

The LLM agent architecture was adapted from Dr. Roman's sample code from the course lab assignment.

---

## Author

**Eric Ortega Rodriguez** · Duke University · CYBERSEC 520 · Spring 2026
GitHub: [github.com/ericiortega](https://github.com/ericiortega)
