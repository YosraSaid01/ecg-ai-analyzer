# 🫀 ECG AI Analyzer

<p align="center">
Signal Processing · Clinical Feature Extraction · Explainable Rhythm Analysis
</p>

<p align="center">
<img src="https://img.shields.io/badge/Domain-Medical%20Signal%20Processing-red?style=for-the-badge" />
<img src="https://img.shields.io/badge/Type-Explainable%20AI-blue?style=for-the-badge" />
<img src="https://img.shields.io/badge/Data-MIT--BIH%20Arrhythmia-green?style=for-the-badge" />
</p>

---

## 📖 Project Overview

A modular, production-style ECG analysis pipeline that transforms raw signals into **clinically interpretable insights**.

Built on the **MIT-BIH Arrhythmia Database (PhysioNet)**, this project demonstrates how to go from:

➡️ Raw ECG signal → Signal processing → Feature extraction → Clinical interpretation

The system performs:

- R-peak detection  
- Heart rate & HRV computation  
- Rule-based abnormality detection  
- Evaluation against expert annotations  
- Generation of a structured clinical narrative  

⚡ **Key idea:** bridge the gap between **signal processing** and **clinical reasoning**, without deep learning or external APIs.

---

## 🧠 Key Features

- 📈 **Robust preprocessing**
  - Bandpass filtering (0.5–40 Hz)
  - Baseline removal
  - Normalization

- ❤️ **R-peak detection**
  - Adaptive detection using `scipy.signal.find_peaks`

- 📊 **Clinical feature extraction**
  - Heart rate (HR)
  - RR intervals
  - HRV metrics (SDNN, RMSSD, pNN50)

- ⚠️ **Explainable abnormality detection**
  - Tachycardia / Bradycardia
  - Irregular rhythm detection
  - Fully rule-based → interpretable

- 🧪 **Evaluation against expert annotations**
  - Precision / Recall / F1-score
  - AAMI-style matching

- 📝 **Clinical narrative generation**
  - Structured interpretation
  - Deterministic (no LLM / no API)

- 📉 **Visualization**
  - ECG + detected peaks
  - Heart rate evolution
  - Highlighted abnormal segments

---

## ⚙️ Pipeline Overview
MIT-BIH Record
│
▼
┌──────────────┐
│ data_loader │ Load ECG signal + expert annotations (WFDB)
└──────┬───────┘
▼
┌──────────────┐
│ preprocess │ Bandpass (0.5–40 Hz) + normalization
└──────┬───────┘
▼
┌──────────────┐
│ peaks │ R-peak detection (adaptive)
└──────┬───────┘
▼
┌──────────────┐
│ features │ HR, RR intervals, HRV metrics
└──────┬───────┘
▼
┌──────────────┐
│ rules │ Clinical abnormality detection
└──────┬───────┘
▼
┌──────────────┐
│ evaluate │ Metrics vs expert annotations
└──────┬───────┘
▼
┌──────────────┐
│ visualize │ ECG plots + abnormal segments
└──────┬───────┘
▼
┌──────────────┐
│ explainer │ Clinical narrative (rule-based)
└──────────────┘

---

## 🗂️ Project Structure
ecg-ai-analyzer/
│
├── app/
│ ├── main.py # Full pipeline orchestrator (CLI)
│ ├── data_loader.py # MIT-BIH loading (WFDB)
│ ├── preprocess.py # Filtering & normalization
│ ├── peaks.py # R-peak detection
│ ├── features.py # HR & HRV extraction
│ ├── rules.py # Abnormality detection
│ ├── evaluate.py # Performance metrics
│ ├── visualize.py # ECG plotting
│ └── llm_explainer.py # Clinical explanation generator
│
├── data/
│ └── mitdb/ # MIT-BIH dataset files
│
├── notebooks/ # Exploration notebooks
├── requirements.txt
└── README.md
---

🛠️ Installation
git clone https://github.com/your-username/ecg-ai-analyzer.git
cd ecg-ai-analyzer

python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
📥 Download Dataset
python -c "import wfdb; wfdb.dl_database('mitdb', 'data/mitdb')"

Or manually from:
https://physionet.org/content/mitdb/1.0.0/

▶️ Usage
Run full pipeline
python app/main.py
Example with parameters
python app/main.py --record 201 --start 10 --duration 20 --save-plots
Argument	Description	Default
--record	Record ID (e.g. 100, 201)	100
--start	Start time (s)	0
--duration	Window duration (s)	10
--save-plots	Save outputs	False
🔬 Signal Processing Details
Step	Method
Filtering	Butterworth bandpass (0.5–40 Hz)
Baseline removal	High-pass filtering
Normalization	Z-score
Peak detection	Adaptive prominence (find_peaks)
📊 Evaluation Methodology
Reference: Expert cardiologist annotations
Tolerance window: ±150 ms
Matching: Greedy nearest-neighbor
Metrics:
Precision
Recall
F1-score
Temporal offset error
🧬 Dataset

MIT-BIH Arrhythmia Database (PhysioNet)

48 annotated ECG recordings
360 Hz sampling frequency
Clinical-grade annotations
Gold standard for arrhythmia research

📚 References:

Moody & Mark, IEEE EMBS (2001)
Goldberger et al., Circulation (2000)
🎯 Why This Project Matters

This project showcases:

Biomedical signal processing expertise
Clinical reasoning from data
Explainable AI (rule-based interpretation)
Reproducible medical pipelines
End-to-end system design (data → insights)

💡 Similar principles are used in:

ECG monitoring systems
Clinical decision support tools
Medical device software pipelines

🚀 Future Improvements

Deep learning (CNN / LSTM for arrhythmia classification)
Streamlit web interface (interactive ECG analysis)
Multi-lead ECG processing
Real-time monitoring pipeline
Advanced HRV (frequency-domain, nonlinear metrics)
⚠️ Disclaimer

This project is intended for research, education, and demonstration purposes only.
It is not a medical device and must not be used for clinical diagnosis.

👩‍💻 Author

Yosra Said
Biomedical Engineer — Medical Imaging & AI

GitHub: https://github.com/YosraSaid01
LinkedIn: (add your link)
⭐ If you like this project

Give it a star ⭐ and feel free to connect!
