# 🫀 ECG AI Analyzer

> **Signal Processing · Clinical Feature Extraction · Explainable Rhythm Analysis**

![Domain](https://img.shields.io/badge/Domain-Medical%20Signal%20Processing-red?style=for-the-badge)
![Type](https://img.shields.io/badge/Type-Explainable%20AI-blue?style=for-the-badge)
![Data](https://img.shields.io/badge/Data-MIT--BIH%20Arrhythmia-green?style=for-the-badge)

---

## 📖 Overview

A modular, production-style ECG analysis pipeline that transforms raw signals into **clinically interpretable insights**, built on the **MIT-BIH Arrhythmia Database (PhysioNet)**.

```
Raw ECG signal → Signal processing → Feature extraction → Clinical interpretation
```

The system performs:

- R-peak detection
- Heart rate & HRV computation
- Rule-based abnormality detection
- Evaluation against expert annotations
- Generation of a structured clinical narrative

> ⚡ **Key idea:** bridge the gap between **signal processing** and **clinical reasoning** — without deep learning or external APIs.

---

## 🧠 Features

| Category | Details |
|---|---|
| 📈 **Preprocessing** | Bandpass filtering (0.5–40 Hz), baseline removal, normalization |
| ❤️ **R-peak detection** | Adaptive detection via `scipy.signal.find_peaks` |
| 📊 **Clinical features** | Heart rate, RR intervals, HRV metrics (SDNN, RMSSD, pNN50) |
| ⚠️ **Abnormality detection** | Tachycardia / Bradycardia / irregular rhythm — fully rule-based |
| 🧪 **Evaluation** | Precision / Recall / F1-score, AAMI-style annotation matching |
| 📝 **Clinical narrative** | Structured, deterministic interpretation (no LLM / no API) |
| 📉 **Visualization** | ECG + detected peaks, heart rate evolution, highlighted abnormal segments |

---

## ⚙️ Pipeline

```
MIT-BIH Record
      │
      ▼
┌─────────────┐
│ data_loader │  Load ECG signal + expert annotations (WFDB)
└──────┬──────┘
       ▼
┌─────────────┐
│  preprocess │  Bandpass (0.5–40 Hz) + normalization
└──────┬──────┘
       ▼
┌─────────────┐
│    peaks    │  R-peak detection (adaptive)
└──────┬──────┘
       ▼
┌─────────────┐
│   features  │  HR, RR intervals, HRV metrics
└──────┬──────┘
       ▼
┌─────────────┐
│    rules    │  Clinical abnormality detection
└──────┬──────┘
       ▼
┌─────────────┐
│   evaluate  │  Metrics vs expert annotations
└──────┬──────┘
       ▼
┌─────────────┐
│  visualize  │  ECG plots + abnormal segments
└──────┬──────┘
       ▼
┌─────────────┐
│  explainer  │  Clinical narrative (rule-based)
└─────────────┘
```

---

## 🗂️ Project Structure

```
ecg-ai-analyzer/
│
├── app/
│   ├── main.py            # Full pipeline orchestrator (CLI)
│   ├── data_loader.py     # MIT-BIH loading (WFDB)
│   ├── preprocess.py      # Filtering & normalization
│   ├── peaks.py           # R-peak detection
│   ├── features.py        # HR & HRV extraction
│   ├── rules.py           # Abnormality detection
│   ├── evaluate.py        # Performance metrics
│   ├── visualize.py       # ECG plotting
│   └── llm_explainer.py   # Clinical explanation generator
│
├── data/
│   └── mitdb/             # MIT-BIH dataset files
│
├── notebooks/             # Exploration notebooks
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/ecg-ai-analyzer.git
cd ecg-ai-analyzer

python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

### 📥 Download Dataset

```bash
python -c "import wfdb; wfdb.dl_database('mitdb', 'data/mitdb')"
```

Or download manually from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/).

---

## ▶️ Usage

**Run the full pipeline:**

```bash
python app/main.py
```

**With parameters:**

```bash
python app/main.py --record 201 --start 10 --duration 20 --save-plots
```

| Argument | Description | Default |
|---|---|---|
| `--record` | Record ID (e.g. `100`, `201`) | `100` |
| `--start` | Start time (s) | `0` |
| `--duration` | Window duration (s) | `10` |
| `--save-plots` | Save output plots | `False` |

---

## 🔬 Signal Processing

| Step | Method |
|---|---|
| Filtering | Butterworth bandpass (0.5–40 Hz) |
| Baseline removal | High-pass filtering |
| Normalization | Z-score |
| Peak detection | Adaptive prominence (`find_peaks`) |

---

## 📊 Evaluation Methodology

- **Reference:** Expert cardiologist annotations
- **Tolerance window:** ±150 ms
- **Matching:** Greedy nearest-neighbor
- **Metrics:** Precision · Recall · F1-score · Temporal offset error

---

## 🧬 Dataset

**MIT-BIH Arrhythmia Database** — [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)

- 48 annotated ECG recordings
- 360 Hz sampling frequency
- Clinical-grade annotations
- Gold standard for arrhythmia research

**References:**
- Moody & Mark, *IEEE EMBS* (2001)
- Goldberger et al., *Circulation* (2000)

---

## 🎯 Why This Project Matters

This project demonstrates:

- **Biomedical signal processing** expertise
- **Clinical reasoning** from raw data
- **Explainable AI** through rule-based interpretation
- **Reproducible medical pipelines**
- **End-to-end system design** — from raw signal to structured insight

Similar principles are used in ECG monitoring systems, clinical decision support tools, and medical device software pipelines.

---

## 🚀 Future Improvements

- [ ] Deep learning (CNN / LSTM for arrhythmia classification)
- [ ] Streamlit web interface for interactive ECG analysis
- [ ] Multi-lead ECG processing
- [ ] Real-time monitoring pipeline
- [ ] Advanced HRV (frequency-domain, nonlinear metrics)

---

## ⚠️ Disclaimer

> This project is intended for **research, education, and demonstration purposes only**.
> It is **not a medical device** and must not be used for clinical diagnosis.

---

## 👩‍💻 Author

**Yosra Said** — Biomedical Engineer · Medical Imaging & AI

[![GitHub](https://img.shields.io/badge/GitHub-YosraSaid01-black?style=flat-square&logo=github)](https://github.com/YosraSaid01)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/your-link)

---

*If this project was useful, give it a ⭐ and feel free to connect!*
