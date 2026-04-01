# 🧠 Smart Data Analyzer — No-Code AI Tool

> **Upload any CSV → Get instant AI-powered analysis, charts, and a chat interface.  
> No code required.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit)
![Claude AI](https://img.shields.io/badge/Claude-AI--Powered-purple?logo=anthropic)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 What Is This?

**Smart Data Analyzer** is a beginner-friendly, production-quality web app that lets anyone analyze a CSV dataset — no Python or statistics knowledge required.

Drop in a file, and within seconds you get:

- ✅ Automatic **summary statistics** (mean, median, std, missing values)
- ✅ Beautiful **interactive charts** (histograms, bar charts, correlation heatmap, scatter plots)
- ✅ **AI-generated insights** explained in plain English
- ✅ A **chat interface** to ask questions about your data naturally

---

## ✨ Features

| Feature | Description |
|---|---|
| 📂 **File Upload** | Drag & drop CSV files up to 200MB |
| 📊 **Data Overview** | Column info, data types, missing values, duplicates |
| 📐 **Statistics** | Mean, median, std dev, skewness, kurtosis |
| 📈 **Visualizations** | Histograms with KDE, bar charts, correlation heatmap, scatter with trend line |
| 🤖 **AI Insights** | Claude-powered natural language analysis |
| 💬 **Chat Interface** | Ask questions about your data conversationally |
| ⬇️ **Export** | Download AI insights as a `.txt` file |
| ⭐ **Smart Columns** | Auto-detects high-importance columns |

---

## 🖥️ Screenshots

### Upload Screen
> _[Screenshot placeholder — Upload & Dataset Preview]_

### Data Overview
> _[Screenshot placeholder — Column Info & Statistics]_

### Visualizations
> _[Screenshot placeholder — Charts Dashboard]_

### AI Insights
> _[Screenshot placeholder — Claude-Generated Insights]_

### Chat Interface
> _[Screenshot placeholder — Chat with Data]_

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- An [Anthropic API key](https://console.anthropic.com) (free tier available)

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/smart-data-analyzer.git
cd smart-data-analyzer
```

### 2. Create a virtual environment

```bash

# Activate it:
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

```bash
cp .env.example .env
```

Open `.env` and add your key:

```
ANTHROPIC_API_KEY=sk-ant-...your-key-here...
```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open automatically at **http://localhost:8501** 🎉

---

## 📁 Project Structure

```
smart-data-analyzer/
│
├── app.py                   # Main Streamlit application & UI
│
├── utils/
│   ├── __init__.py          # Makes utils a Python package
│   ├── data_processor.py    # Data loading, cleaning, statistics
│   ├── visualizer.py        # All chart generation (Matplotlib/Seaborn)
│   └── ai_helper.py         # Anthropic Claude API integration
│
├── requirements.txt         # Python dependencies
├── .env.example             # API key template (copy to .env)
├── .gitignore               # Files to ignore in version control
└── README.md                # This file
```

### What each file does

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI, routing, session state, page rendering |
| `utils/data_processor.py` | Loads CSV, computes stats, builds AI summary text |
| `utils/visualizer.py` | Matplotlib/Seaborn charts with dark theme |
| `utils/ai_helper.py` | Calls Claude API for insights and chat |

---

## 🔧 Configuration

All configuration is via environment variables in `.env`:

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ Yes | Your Anthropic API key |

---

## 🧪 Testing with Sample Data

Don't have a CSV handy? Try these free datasets:

- [Titanic](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) — passenger survival data
- [Iris](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv) — flower measurements
- [Tips](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv) — restaurant tips

Save any of these as a `.csv` file and upload it to the app.

---

## 🔮 Future Improvements

- [ ] Support for Excel (`.xlsx`) and JSON files
- [ ] Export full report as PDF
- [ ] Time series detection and trend charts
- [ ] Column-level AI descriptions (auto-label cryptic columns)
- [ ] Data cleaning suggestions (drop nulls, fix types)
- [ ] Multi-file upload and comparison
- [ ] Shareable report links
- [ ] Google Sheets integration

---

## 🛠️ Built With

- [Streamlit](https://streamlit.io) — Web UI framework
- [Pandas](https://pandas.pydata.org) — Data manipulation
- [Matplotlib](https://matplotlib.org) + [Seaborn](https://seaborn.pydata.org) — Visualizations
- [Anthropic Claude](https://anthropic.com) — AI insights & chat
- [python-dotenv](https://pypi.org/project/python-dotenv/) — Environment management

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙋 Contributing

Pull requests welcome! For major changes, open an issue first.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push and open a PR

---

*Built with ❤️ as a portfolio project. Powered by Claude AI.*
