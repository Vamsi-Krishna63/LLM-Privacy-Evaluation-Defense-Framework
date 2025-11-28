# LLM-Privacy-Evaluation-Defense-Framework (LLM-PBE)

A comprehensive, modular framework for evaluating privacy risks in Large Language Models (LLMs). This project implements multiple privacy attack strategies, defense mechanisms, evaluation metrics, and analysis workflows, enabling researchers and practitioners to assess and mitigate sensitive data leakage in LLM systems.

## 📌 Overview

LLM-PBE provides an end-to-end pipeline for:

- 🔍 Privacy Attack Simulation — Membership inference, prompt leakage, data extraction, jailbreak attacks, etc.
- 🛡️ Defense Application — Prompt sanitization, output filtering, contextual defense strategies.
- 📊 Evaluation & Metrics — Privacy leakage scores, attack success rate, defense effectiveness.
- 📁 Experiment Automation — Unified runners for reproducible experiments.
- 📈 Visualization — Result plots, logs, and analytics for research reporting.

This framework helps organizations and researchers validate privacy guarantees before deploying LLMs in sensitive domains like healthcare, finance, and user-generated content platforms.

## 📂 Repository Structure

```
LLM-Privacy-Evaluation-Defense-Framework/
│
├── attacks/               # All implemented privacy attacks
├── defenses/              # Defense strategies and mitigation techniques
├── metrics/               # Evaluation metrics & scoring modules
├── models/                # Model wrappers & configuration (API / local)
├── data/                  # Input data samples or test sets
├── scripts/               # Automated experiment runners
├── notebooks/             # Jupyter notebooks for analysis & visualization
├── generations/           # Sample LLM outputs collected during attacks
├── results/               # Logs, plots, CSV metrics from experiments
│
└── AttackDemo.py          # Simple example demonstrating attack execution
```

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/Vamsi-Krishna63/LLM-Privacy-Evaluation-Defense-Framework.git
cd LLM-Privacy-Evaluation-Defense-Framework
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

Python 3.8+ is recommended.

## ▶️ Running the Framework

### Run the default attack demo
```bash
python AttackDemo.py
```

### Use other experiment scripts
Run experiments from the `scripts/` directory:
```bash
python scripts/run_privacy_evaluation.py
```

## 🔬 Supported Attacks

- Membership Inference Attacks (MIA)
- Prompt Leakage Attacks
- Data Extraction via Repeated Prompting
- Jailbreak-based Privacy Violations
- Sensitive Information Reconstruction Attacks

## 🛡️ Defense Mechanisms

- Prompt Scrubbing / Redaction
- Output Filtering
- Heuristic-based Safety Layers
- Contextual Defense (LLM-as-a-Guardrail)

## 📈 Evaluation Metrics

- Attack Success Rate (ASR)
- Privacy Leakage Score
- Defense Effectiveness Score
- Precision, Recall, FPR
- Confidence Gap Analysis

Results are saved automatically in the `results/` directory.

## 🧪 Visualization & Analysis

```bash
jupyter notebook notebooks/
```

Includes heatmaps, breakdowns, and graphical plots for analysis.

## 🏗️ Extending the Framework

- Add attacks → `attacks/`
- Add defenses → `defenses/`
- Add metrics → `metrics/`

Modular design makes extension easy.

## 🤝 Contributing

Contributions are welcome!  
Submit PRs for attacks, defenses, evaluation modules, or documentation improvements.

## 📝 License

MIT License (recommended – update based on your preference)

## 👨‍💻 Author

**Vamsi Krishna**  
GitHub: https://github.com/Vamsi-Krishna63
