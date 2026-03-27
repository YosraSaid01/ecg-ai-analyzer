# ECG AI Analyzer

**Signal Processing · Rhythm Detection · Clinical Explanation**

A modular, production-quality ECG analysis pipeline built on the
[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
from PhysioNet.  The project applies standard biomedical signal processing
techniques to detect R-peaks, compute clinically interpretable features,
flag rhythm abnormalities with rule-based logic, evaluate detection
accuracy against expert annotations, and generate a structured clinical
narrative — all without deep learning or external API calls.

---

## Pipeline Overview

```
MIT-BIH Record
     │
     ▼
 ┌──────────────┐
 │  data_loader  │  Load ECG signal + expert annotations (WFDB)
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │  preprocess   │  Bandpass filter (0.5–40 Hz) + normalization
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │    peaks      │  R-peak detection (scipy.signal.find_peaks, adaptive)
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │   features    │  RR intervals, heart rate, HRV (SDNN, RMSSD, pNN50)
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │    rules      │  Tachycardia / bradycardia / irregular rhythm flags
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │   evaluate    │  TP / FP / FN / Precision / Recall / F1 vs annotations
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │  visualize    │  Matplotlib plots: ECG, peaks, HR, abnormal segments
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │ llm_explainer │  Rule-based clinical narrative (no API calls)
 └──────────────┘
```

---

## Project Structure

```
ecg-ai-analyzer/
│
├── app/
│   ├── main.py              # Full pipeline orchestrator (CLI)
│   ├── data_loader.py        # MIT-BIH record + annotation loader
│   ├── preprocess.py         # Baseline removal, bandpass, normalization
│   ├── peaks.py              # R-peak detection
│   ├── features.py           # Clinical feature extraction
│   ├── rules.py              # Rule-based abnormality detection
│   ├── evaluate.py           # Detection performance evaluation
│   ├── visualize.py          # ECG plotting
│   └── llm_explainer.py      # Clinical narrative generator
│
├── data/
│   └── mitdb/                # MIT-BIH database files (*.dat, *.hea, *.atr)
│
├── notebooks/                # Exploratory Jupyter notebooks
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ecg-ai-analyzer.git
cd ecg-ai-analyzer
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the MIT-BIH Arrhythmia Database

The pipeline expects WFDB-format files in `data/mitdb/`.  You can
download them automatically with the WFDB library:

```bash
python -c "import wfdb; wfdb.dl_database('mitdb', 'data/mitdb')"
```

This downloads all 48 half-hour records (~100 MB).  Each record
consists of three files (e.g. `100.dat`, `100.hea`, `100.atr`).

Alternatively, download manually from:
https://physionet.org/content/mitdb/1.0.0/

---

## Usage

### Run the full pipeline

```bash
python app/main.py
```

This loads record **100** by default, runs all processing steps, shows
plots, and prints the clinical explanation.

### Command-line options

```bash
python app/main.py --record 201 --start 10 --duration 20 --save-plots
```

| Flag           | Description                              | Default |
|----------------|------------------------------------------|---------|
| `--record`     | MIT-BIH record ID (e.g. `100`, `201`)    | `100`   |
| `--start`      | Visualization start time (seconds)       | `0.0`   |
| `--duration`   | Visualization window length (seconds)    | `10.0`  |
| `--save-plots` | Save plot images to `outputs/` directory | off     |

### Run individual modules

Each module can be executed independently for testing:

```bash
python app/data_loader.py       # Test data loading
python app/preprocess.py        # Test filtering on synthetic signal
python app/peaks.py             # Test peak detection on synthetic signal
python app/features.py          # Test feature computation
python app/rules.py             # Test rule engine
python app/evaluate.py          # Test evaluation metrics
python app/llm_explainer.py     # Test narrative generation
```

---

## Signal Processing Details

| Stage             | Method                                  | Parameters           |
|-------------------|-----------------------------------------|----------------------|
| Baseline removal  | Butterworth high-pass (zero-phase)      | 0.5 Hz, order 4     |
| Bandpass filter   | Butterworth bandpass (zero-phase)       | 0.5–40 Hz, order 4  |
| Power-line notch  | IIR notch filter (optional)             | 50/60 Hz, Q = 30    |
| Normalization     | Z-score (zero mean, unit variance)      | —                    |
| R-peak detection  | `scipy.signal.find_peaks` + refinement  | Adaptive prominence  |

---

## Evaluation

Detection accuracy is measured against expert annotations using the
AAMI EC57 methodology:

- **Tolerance window**: ±150 ms (configurable)
- **Matching**: greedy one-to-one (closest-first)
- **Metrics**: Precision, Recall, F1-score, mean/std offset

---

## Dataset

**MIT-BIH Arrhythmia Database** (PhysioNet)

- 48 half-hour two-channel ambulatory ECG recordings
- 47 subjects studied at BIH Arrhythmia Laboratory (1975–1979)
- Digitized at 360 samples/second, 11-bit resolution over 10 mV range
- Expert-annotated beat-by-beat by two or more cardiologists

**Citation:**

> Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
> IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).

> Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet:
> Components of a new research resource for complex physiologic signals.
> Circulation 101(23):e215-e220 (2000).

---

## Future Improvements

- **Deep learning model** — Train a CNN or LSTM for beat classification
  using the annotated beat types (N, V, A, etc.)
- **Streamlit interface** — Interactive web dashboard for uploading and
  analyzing ECG recordings in the browser
- **LLM integration** — Connect to a language model API for richer,
  context-aware clinical explanations
- **Multi-lead analysis** — Extend preprocessing and detection to both
  channels (MLII + V1/V5)
- **Real-time streaming** — Sliding-window processing for continuous
  monitoring scenarios
- **Extended HRV analysis** — Frequency-domain (LF/HF ratio) and
  non-linear metrics (Poincaré plots, sample entropy)

---

## License

This project is provided for educational and research purposes.
The MIT-BIH Arrhythmia Database is available under the
[PhysioNet Open Data License](https://physionet.org/content/mitdb/1.0.0/).
