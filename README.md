# 🌍 Job Market Intelligence & Career Advisor

> **Ironhack Final Project — Data Science & Machine Learning**

A dashboard that answers **"How much should I earn, and where are the jobs?"**
It combines **salary analysis** (EDA), **salary prediction** (Machine Learning),
an **AI career coach** (RAG chatbot), and a **live job finder** across 7 regions.

![App: 4 tabs](https://img.shields.io/badge/tabs-4-blue)

---

## 🚀 Quick start

### 1. Install everything

Open a **terminal** in this project folder and run:

```bash
bash setup.sh
```

This creates a private environment (`.venv`) and installs all the libraries.
It automatically installs the **CPU-only** version of `torch`, so it works on
any computer — **with or without an NVIDIA GPU** (most laptops have none).

> **Windows?** Don't worry — run these three commands instead in PowerShell:
> ```powershell
> python -m venv .venv
> .\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
> .\.venv\Scripts\pip install -r requirements.txt
> ```

### 2. API keys & the AI Career Coach

Create a file named `.env` in this folder (same place as `README.md`).

The **AI Career Coach** runs on a **local model by default — Qwen3-Coder-Next
served by llama.cpp — so it needs no API key**. If you prefer the cloud, you can
switch it back to the DeepSeek API instead:

```env
# AI Career Coach — pick ONE option:
#   A) local model (llama.cpp, no real key — "local" is just a placeholder):
DEEPSEEK_API_KEY=local
DEEPSEEK_BASE_URL=http://localhost:8080/v1
DEEPSEEK_MODEL=qwen3-coder-next
#   B) cloud DeepSeek API (uncomment to use instead):
# DEEPSEEK_API_KEY=sk-...                  # → https://platform.deepseek.com
# DEEPSEEK_BASE_URL=https://api.deepseek.com
# DEEPSEEK_MODEL=deepseek-chat

TAVILY_API_KEY=tvly-...     # for the Live Job Finder   → https://tavily.com
```

> To use option **A**, run your local model with llama.cpp's `llama-server` on
> `http://localhost:8080` and keep it running while you use Tab 3.
> ⚠️ Never share this file — it holds your private keys.

### 3. Start the app

```bash
.venv/bin/streamlit run app/streamlit_app.py
```

Your browser opens `http://localhost:8501` with the dashboard. That's it! 🎉

---

## 📁 Project layout

| Folder / file | What it is |
|---|---|
| `app/` | ⭐ The dashboard — `streamlit_app.py` is the app you run |
| `src/` | The "engine room" — Python modules the app uses |
| `notebooks/` | Step-by-step Jupyter notebooks (explore → ML → RAG) |
| `data/` | The data: `raw/` (downloaded) and `processed/` (cleaned) |
| `sql/` | Database schemas (how jobs/applications are stored) |
| `visuals/` | Chart-export tool + Tableau guide (optional extras) |
| `models/` | Trained ML models (created when you run the notebooks) |
| `setup.sh` | One-command installer (Linux/macOS) |
| `requirements.txt` | List of libraries to install |
| `.env` | Your secret API keys + optional local-model settings (you create this) |
| `.venv/` | The private environment (created by setup.sh — leave it) |

### What each `src/` module does

| Module | Job |
|---|---|
| `data_loader.py` | Reads & cleans the salary data |
| `preprocessing.py` | Prepares data for machine learning |
| `models.py` | Trains & evaluates the ML models |
| `rag_engine.py` | Powers the AI Career Coach (retrieval + local Qwen3 or DeepSeek API) |
| `tavily_job_finder.py` | Live job search across 7 regions |
| `applications.py` | Saves jobs & sends applications (SQLite) |
| `utils.py` | Shared settings & API keys |

---

## 🧠 The 4 tabs

| Tab | What it does |
|---|---|
| **📊 Market Explorer** | Interactive charts — salary by experience, company size, field |
| **💰 Salary Predictor** | Pick role + experience + location → estimated salary |
| **🤖 AI Career Coach** | Chat with a RAG assistant backed by real salary data (local Qwen3 — no key needed; cloud DeepSeek optional) |
| **🌍 Live Job Finder** | Search live DS/ML/DL + Aerospace jobs in 7 regions (needs Tavily key) |

---

## 📓 The notebooks

Run them if you want to re-generate the data, charts, and models from scratch:

| # | Notebook | What it does |
|---|---|---|
| 1 | `01_data_exploration.ipynb` | Downloads data, cleans it, creates `data/processed/*.csv` + charts |
| 2 | `02_ml_modeling.ipynb` | Trains ML models, saves them to `models/` |
| 3 | `03_rag_build.ipynb` | Builds the ChromaDB vector store for the AI Coach |

> The app works out of the box because `data/processed/*.csv` is already
> included. The notebooks are there so you can see *how* it was built.

---

## 🛠 Tech stack

| Layer | Technology |
|---|---|
| Data | Pandas, NumPy |
| Charts | Matplotlib, Seaborn, Plotly |
| ML | Scikit-learn, XGBoost, LightGBM |
| RAG / AI | Qwen3-Coder-Next (local via llama.cpp) or DeepSeek API, LangChain, ChromaDB, sentence-transformers |
| App | Streamlit |
| Live search | Tavily API |

---

*built by enthusiasts*
