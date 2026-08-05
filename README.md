# 🌍 Job Market Intelligence & Career Advisor — v4

> **Ironhack Final Project — Unified End-to-End Data Science**

A combined platform merging **all data jobs analysis (v2)** with **DS/ML/DL-specific insights (v3)** plus a **Tavily-powered live job finder** scanning 6 global regions.

```bash
streamlit run app/streamlit_app.py
```

---

## 🎯 What v4 Adds

| Feature | Description |
|---|---|
| **v2/v3 Toggle** | Radio button switches between "All Jobs" and "DS/ML/DL Only" |
| **🌍 Live Job Finder** | Tavily searches 6 regions for real-time DS/ML/DL jobs |
| **Unified Data** | Both datasets (555 + 411 rows) in one project |
| **Job Export** | Download found jobs as CSV |

---

## 📊 4 Tabs

| Tab | What It Does |
|---|---|
| **Market Explorer** | Charts with v2/v3 toggle — salary by experience, company size, DS/ML field |
| **Salary Predictor** | Select role + experience + location → salary estimate |
| **AI Career Coach** | Chat with DeepSeek RAG (answers backed by real data) |
| **🌍 Live Job Finder** | Tavily searches Europe, Middle East, China, Russia, South America, East Asia |

---

## 🌍 Live Job Finder Regions

| Region | Search Focus |
|---|---|
| **Europe** | LinkedIn, Indeed — DS/ML/AI roles |
| **Middle East** | Dubai, UAE, Saudi Arabia, Qatar |
| **China** | Beijing, Shanghai — zhaopin.com, 51job.com |
| **Russia** | Moscow — DS/ML/DL vacancies |
| **South America** | Brazil, Argentina, Chile, Colombia |
| **East Asia** | Japan, Korea, Singapore, Taiwan |

---

## 🧱 Tech Stack

| Layer | Technology |
|---|---|
| Data | Pandas, Kaggle (DS Salaries) |
| Web Search | **Tavily** (live job collection) |
| ML | Scikit-learn, XGBoost, LightGBM |
| Gen AI / RAG | **DeepSeek**, LangChain, ChromaDB, HuggingFace |
| App | **Streamlit**, Plotly |

---

## 🚀 Quick Start

```bash
cd project-ds-end-to-end-v4
pip install -r requirements.txt

# Copy .env with API keys
cp ../project-ds-end-to-end-v2/.env .env

# Build data
python build_data.py

# Launch
streamlit run app/streamlit_app.py
```

## 🔑 Required API Keys (in .env)

```
DEEPSEEK_API_KEY=sk-...     # For RAG chatbot
TAVILY_API_KEY=tvly-...     # For live job finder
```
