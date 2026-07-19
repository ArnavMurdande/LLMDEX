# 📈 LLMDEX: The Ultimate LLM Benchmark Intelligence Dashboard
### 📊 The Live Analytics Dashboard for Large Language Models  

[![GitHub Repository](https://img.shields.io/badge/GitHub-ArnavMurdande%2FLLMDEX-blue?logo=github)](https://github.com/ArnavMurdande/LLMDEX)
[![Platform](https://img.shields.io/badge/Platform-Web-success)]()
[![Stack](https://img.shields.io/badge/Stack-Python%20%7C%20Static%20Hosting-orange)]()
[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)]()

🌐 **Live Platform:** https://llmdex.onrender.com/  
📂 **Repository:** https://github.com/ArnavMurdande/LLMDEX  

> Think **Bloomberg Terminal, but for AI Models.**

LLMDEX is a public analytics and benchmarking platform that transforms the complete expanded Artificial Analysis model leaderboard into clear performance, coding, value, and efficiency views.
It is **not** an AI model hosting or inference platform — it is a **data intelligence and benchmarking hub**.

The goal is simple:  
Help individuals, AI enthusiasts, developers, and businesses **identify the right model for their specific use case** using real, benchmark-backed analytics.

---

## 🚀 Overview

The LLM ecosystem evolves rapidly. New models launch frequently, benchmarks vary across platforms, and pricing structures change often.

LLMDEX solves this fragmentation by:

- Capturing every expanded Artificial Analysis leaderboard field
- Preserving the authoritative Artificial Analysis performance order
- Calculating a proprietary **Efficiency Score**  
- Tracking model evolution across generations  
- Providing AI-assisted model recommendations  

It acts as a **central intelligence layer** for AI model decision-making.

---

## 🎯 Core Objectives

LLMDEX enables users to:

- 📈 View the latest LLM benchmarks from public leaderboards  
- 🧠 Compare intelligence, reasoning, coding, and multimodal capabilities  
- 💰 Evaluate pricing vs performance using efficiency metrics  
- 🧬 Track model family evolution and historical growth  
- 🤖 Use an AI advisor bot to select the best model for their use case  

---

## 🧩 Key Features

### 1️⃣ Live Benchmark Aggregation

- Scrapes all current models and expanded columns from Artificial Analysis
- Standardized dataset format for fair comparisons  
- Downloadable datasets for research and BI tools  

---

### 2️⃣ Signature Efficiency Score

```
Efficiency Score = Intelligence Score ÷ Cost per Token
```

This metric helps identify:

- Most cost-effective models  
- Best ROI for production systems  
- High-value performers hidden behind hype  

---

### 3️⃣ Comprehensive Model Metrics

Each tracked model includes:

- Model Name  
- Provider  
- Intelligence Score  
- Coding Score  
- Reasoning Score  
- Multimodal Score  
- Cost per Token  
- Context Window  
- Latency  
- Modality Support (Text / Image / Audio / Video)  
- Benchmark Source  

---

### 4️⃣ Model Family Evolution Tracking

LLMDEX visualizes:

- Model lineage  
- Generational improvements  
- Historical growth trends  
- Performance delta between predecessor models  

This allows users to understand **trajectory**, not just static scores.

---

### 5️⃣ Business Intelligence Dashboards

- Power BI integration  
- Tableau compatibility  
- Downloadable datasets  
- Visual performance comparison tools  

---

## 🌐 Data Sources

Artificial Analysis is the only benchmark source:

- https://artificialanalysis.ai/leaderboards/models

The automated scraper expands the live table, captures every current row and
visible column, validates the snapshot, and stores both typed metrics and the
raw public field/value pairs. The family-history step also caches the release
date published on each model record so progressions remain chronological.

## Automatic Updates

- `.github/workflows/update.yml` refreshes Artificial Analysis data daily,
  rebuilds family history and top-10 model sentiment, validates the publication
  contract, commits changed datasets to `main`, and redeploys Pages.
- Dependabot checks Python packages and GitHub Actions every Monday.
- Patch and minor dependency updates are configured for safe auto-merge; pull
  request CI must pass before protected branches allow the merge.
- Failed scheduled updates open or update one GitHub issue with a direct link
  to the failed workflow, so a stale deployment cannot fail silently.

Optional repository secrets:

- `GEMINI_SENTIMENT_KEY_1` through `GEMINI_SENTIMENT_KEY_4` enable semantic
  sentiment classification. Without them, the workflow uses VADER.
- `X_BEARER_TOKEN` enables X recent-search data through X's official API.
- Reddit and Hacker News collection need no project secret. Repository issues
  and news headlines are excluded from community sentiment and public quotes.

---

## 🏗 Architecture

```
Artificial Analysis expanded leaderboard
        ↓
Python Scraping Pipeline
        ↓
Data Cleaning & Normalization
        ↓
Efficiency Score Calculation
        ↓
Structured Dataset (GitHub)
        ↓
Dashboards (Power BI / Tableau)
        ↓
Static/live Website Hosting
```

---

## 🛠 Tech Stack (100% Free Stack)

| Layer | Technology |
|-------|------------|
| Scraping & Data Pipeline | Python |
| Dataset Storage | GitHub |
| Automation | Cron Jobs |
| Dashboards | Power BI / Tableau |
| Hosting | Render  |
| Local Development | http://localhost:8080 |

---

## ⚙️ Local Setup

```bash
# Clone the repository
git clone https://github.com/ArnavMurdande/LLMDEX.git

# Navigate into project directory
cd LLMDEX

# Run the website and Advisor API
python api_server.py
```

Open in your browser:

```
http://localhost:8080
```

---

## 📊 Platform Philosophy

LLMDEX is built on three principles:

1. **Transparency** — Open data and clear methodology  
2. **Comparability** — Standardized metrics across providers  
3. **Practicality** — Help users choose the right model  

It does not rank models based on hype — only measurable analytics.

---

## 🗺 Scope of Improvements

### 🔜 Leaderboard Expansion

- Add Image Generation tracking  
- Add Video Generation benchmarks  
- Add Music / Audio model tracking  

---

### 🔍 Data Accuracy Improvements

- Improve cross-source validation  
- Refine normalization methodology  

---

### 💬 Community Sentiment Refinement

- Improve credibility verification of sentiment charts  
- Validate public comment sources  
- Enhance model-specific community cards  

---

### 🧬 Model Family History Improvements

- Prevent overpopulation of growth charts  
- Implement year-based sorting  
- Improve generational comparison visuals  

---

### 🚀 Operational Enhancements

- Finalize Power BI integration  
- Finalize Tableau dashboard integration  

---

## 📈 Future Vision

LLMDEX aims to become:

- The **IMDB of AI Models**  
- The **Bloomberg Terminal for LLMs**  
- The **Definitive Benchmark Intelligence Layer for AI**  

As AI models evolve, LLMDEX will evolve with them —  
providing clarity in a rapidly changing ecosystem.

---

## 🤝 Contributions

Contributions are welcome.

If you'd like to:

- Improve scraping pipelines  
- Add new benchmark sources  
- Enhance dashboards  
- Refine normalization logic  

Fork the repository and open a pull request.

---

## 📜 License

MIT License  

---

## 👨‍💻 Author

**Arnav Murdande**  
Computer Engineering | AI & Data Systems  
Mumbai  

---

## ⭐ Support the Project

If you found LLMDEX useful, consider giving the repository a ⭐  
and help grow the AI benchmarking ecosystem.

---

> **LLMDEX — Turning AI performance into measurable intelligence.**
