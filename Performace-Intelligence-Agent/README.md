# Performance Intelligence Agent
### Capstone Project — Data Processing · Machine Learning · Generative AI

A personal AI agent that analyses CrossFit training data from SugarWOD, surfaces patterns using machine learning, and lets me have a natural language conversation with your own data via Claude.

---

## What it does

| Component | What it covers |
|-----------|---------------|
| **Data Processing** | Cleans and normalises SugarWOD CSV exports. Visualises attendance, sentiment, and performance trends. |
| **Machine Learning** | KMeans clustering (workout archetypes) · Isolation Forest (anomaly detection) · Linear Regression (PR forecasting) |
| **Generative AI** | Claude (claude-sonnet) acts as a reasoning agent. It decides which analysis tools to call based on my question, interprets the results, and responds in natural language. |

---

## Project structure

```
├── data/
│   └── workouts.csv          # SugarWOD export (not committed — add your own)
├── analysis/
│   ├── attendance.py         # Consistency, gaps, day-of-week patterns
│   ├── sentiment.py          # Keyword scoring on workout notes
│   ├── performance.py        # RX rate, strength progression, lift correlation
│   └── ml_models.py          # Clustering, anomaly detection, PR forecasting
├── loader.py                 # Data loading and cleaning
├── tools.py                  # Agent tool definitions (wraps analysis modules)
├── agent.py                  # Claude agent loop with tool use
├── app.py                    # Streamlit chat interface
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/helensamara/performance-intelligence-agent.git
cd performance-intelligence-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
echo "ANTHROPIC_API_KEY=sk-your-key-here" > .env

# 4. Add your data
# Export workouts CSV from SugarWOD and place at data/workouts.csv

# 5. Run
streamlit run app.py
```

Then open **http://localhost:8501** in the browser.

---

## Example questions to ask the agent

- *"How consistent have I been training?"*
- *"When is my next PR likely for the Snatch?"*
- *"What do my workout notes reveal about how I feel?"*
- *"Are there any unusual sessions worth looking at?"*
- *"What are my training archetypes?"*

---

## Data

This project uses my personal SugarWOD CSV export (initially, 367 workouts, Oct 2023 – Feb 2026).
The data file is excluded from the repo for privacy. The project works with any SugarWOD export
that includes the standard columns: `date, title, description, best_result_raw, best_result_display,
score_type, barbell_lift, set_details, notes, rx_or_scaled, pr`.

---

## Roadmap

- **Stage 2** — Add Garmin biometric data (HRV, sleep, heart rate)
- **Stage 3** — Add powerlifting program (PDF extraction + actual logging)
- **Stage 4** — Proactive alerts and weekly automated reports
- **Stage 5** — YOLO video analysis of lifting form (separate project)
