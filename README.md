# Encrypted C2 Traffic Detector v3.2

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-success.svg)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**Encrypted C2 Traffic Detector v3.2** is a machine learning-based digital forensic tool developed for the **CIS*6520 – Advanced Digital Forensics & Incident Response** course at the **University of Guelph**.

The project investigates whether encrypted Command-and-Control (C2) traffic can be detected **without decrypting network traffic**, while providing explainable results suitable for digital forensic investigations.

Unlike previous research that focuses primarily on detection accuracy, this project emphasizes:

- Explainable machine learning using SHAP
- External validation on an independent dataset
- SIEM-compatible forensic reporting
- Practical workflow for Canadian law enforcement investigators

---

# Research Objective

Current encrypted malware detection systems often suffer from three practical limitations:

- No explanation of why a connection was classified as malicious
- Limited external validation across independent datasets
- No structured forensic output for operational investigators

This project addresses these limitations by combining:

- XGBoost classification
- SHAP explainability
- Independent CTU-13 validation
- JSON forensic reporting

---

# Project Information

**Course**

CIS*6520 – Advanced Digital Forensics & Incident Response

**University**

University of Guelph

**Semester**

Summer 2026

**Intended Stakeholder**

Royal Canadian Mounted Police (RCMP)
National Cybercrime Coordination Centre (NC3)

---

# Team

| Name | Role |
|-------|------|
| Luther Marni | Machine Learning & Data Analysis |
| Samuel Ghebremeskel | Software Development, Digital Forensics & System Integration |

---

# Version 3.2 Features

- XGBoost encrypted C2 classifier
- SHAP global explainability
- Per-alert forensic explanations
- Independent CTU-13 external validation
- Feature alignment diagnostics
- Dataset shift analysis
- Data quality validation
- PCAP analysis support
- CICFlowMeter integration
- Optional TLS metadata enrichment
- SIEM-compatible JSON reports
- Structured output directory
- Multiple feature-profile experiments
- Threshold-based prediction
- Modular architecture

---

# Project Structure

```
c2_detector_v3/

│
├── c2_detector.py
├── README.md
├── requirements.txt
│
├── c2detector/
│   ├── cli.py
│   ├── workflows.py
│   ├── modeling.py
│   ├── explain.py
│   ├── reporting.py
│   ├── pcap.py
│   ├── validation.py
│   ├── data.py
│   └── utils.py
│
├── data/
│
├── output/
│
├── tests/
│
└── models/
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/c2-detector.git

cd c2-detector
```

Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it

Linux

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Structure

```
data/

├── cicids2017/
│
│   └── MachineLearningCVE/
│
├── ctu13/
│
│   └── ctu13_flows/
```

---

# Training

Train the model

```bash
python3 c2_detector.py train --ctu-all-malicious
```

Use a different feature profile

```bash
python3 c2_detector.py train \
    --ctu-all-malicious \
    --feature-profile stable-core
```

Available feature profiles

- all-cleaned
- stable-core
- no-active-idle
- no-windows
- no-subflow

---

# Predict Using Flow CSV

```bash
python3 c2_detector.py predict \
    --input suspect_Flow.csv
```

Custom threshold

```bash
python3 c2_detector.py predict \
    --input suspect_Flow.csv \
    --threshold 0.75
```

---

# Predict Using PCAP

```bash
python3 c2_detector.py predict-pcap \
    --input suspect.pcap
```

Optional TLS enrichment

```bash
python3 c2_detector.py predict-pcap \
    --input suspect.pcap \
    --tls-enrich
```

---

# Output Directory

```
output/

├── models/
├── reports/
├── validation/
├── shap/
├── flows/
├── tls/
└── logs/
```

Generated reports include

- Trained model
- Training report
- Forensic JSON report
- SHAP summary plot
- Confusion matrix
- ROC curve
- Feature alignment report
- Feature distribution report
- Data quality report
- CTU probability distribution

---

# Machine Learning Pipeline

```
PCAP
      │
      ▼

CICFlowMeter

      │
      ▼

Flow Features

      │
      ▼

XGBoost Classifier

      │
      ▼

SHAP Explainability

      │
      ▼

JSON Forensic Report
```

---

# Validation Methodology

The model is trained only on the Canadian **CICIDS2017** dataset.

Evaluation consists of two stages.

### Internal Validation

Train/Test split using CICIDS2017.

Metrics reported

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- False Positive Rate
- False Negative Rate

### External Validation

Independent validation using **CTU-13 Scenario 1 (Neris Botnet)**.

Additional reports

- Feature Alignment
- Dataset Shift Analysis
- Prediction Probability Distribution

This evaluates how well the model generalizes to unseen malware traffic.

---

# Explainable AI

Version 3.2 integrates **SHAP (SHapley Additive exPlanations)**.

For every prediction, SHAP identifies which features contributed most to the classification.

Example:

```
Top Indicators

Flow Duration

Destination Port

Init_Win_bytes_forward

Flow IAT Mean

Packet Length Mean
```

---

# JSON Forensic Report

Each investigation produces a SIEM-compatible JSON report containing

- Case metadata
- Model version
- Alert confidence
- SHAP explanations
- Flow information
- Network indicators
- TLS metadata (optional)

---

# Current Limitations

Current limitations include

- Validation uses a malicious-only CTU-13 scenario assumption.
- Cross-dataset performance is affected by dataset shift.
- Different CICFlowMeter versions may produce different feature distributions.
- TLS enrichment currently supplements the flow model rather than serving as model input.

---

# Future Work (Version 4.0)

Planned improvements include

- Direct PCAP processing without external CICFlowMeter dependency
- Automatic CTU-13 label alignment using `capture.labeled`
- Additional malware datasets
- Deep learning comparison (LSTM/Transformer)
- Live network monitoring mode
- Splunk and ELK integration
- Docker deployment
- REST API
- Interactive web dashboard

---

# References

1. Anderson, B., & McGrew, D. (2016). Machine Learning for Encrypted Malware Traffic Classification.
2. Fu, Li & Xu (2023). Graph-Based Encrypted Traffic Detection.
3. Cui et al. (2023). Behaviour-Based Encrypted Malware Detection.
4. CICIDS2017 Dataset, Canadian Institute for Cybersecurity, University of New Brunswick.
5. CTU-13 Botnet Dataset, Czech Technical University.

---

# License

- MIT
