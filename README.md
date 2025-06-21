# University of San Diego  
## AAI-510: Machine Learning Fundamentals and Applications

**Professor**: Bilgenur Baloglu, Ph.D  
**Section**: 4  
**Group**: 3  
**Contributors**:
- Swapnil Patil
- Kevin Hooman
- Dillard Holder

# Patient Appointments Show/No Show Prediction and Analysis

## Overview
This project predicts patient appointment outcomes (show/no-show) and analyzes factors influencing these outcomes using machine learning. It leverages supervised and unsupervised learning, as well as NLP for sentiment and no-show reason analysis.

## Problem Statement / Objective / Use Cases

### Problem Statement
Missed medical appointments (no-shows) negatively impact healthcare efficiency, resource allocation, and patient outcomes. Understanding and predicting no-shows can help clinics optimize scheduling, reduce costs, and improve patient care.

### Objective
- Predict whether a patient will show up for their scheduled appointment using machine learning.
- Analyze key factors (demographics, appointment details, communication, sentiment) influencing no-shows.
- Extract and interpret patient sentiment and reasons for no-shows using NLP techniques.

### Use Cases
- **Clinic Scheduling Optimization:** Proactively identify high-risk no-show appointments and adjust schedules or send reminders.
- **Patient Engagement:** Tailor communication strategies based on predicted risk and sentiment analysis.
- **Resource Planning:** Allocate staff and resources more efficiently by anticipating attendance patterns.
- **Healthcare Analytics:** Provide actionable insights for administrators to reduce no-show rates and improve patient satisfaction.

## Demo

https://smartcareai.streamlit.app/

## Detailed Document

[Notebook](https://swapnilprakashpatil.github.io/aai510_3proj/Final%20Project%20Section4-Team%203.html)

## Project Architecture

```
aai510_3proj/
│
├── data/
│   ├── raw/                # Raw data files
│   ├── processed/          # Processed data files
│   ├── emotion_variations.csv
│   └── negation_patterns.csv
│
├── models/
│   ├── supervised/         # Supervised learning models
│   ├── unsupervised/       # Unsupervised learning models
│   └── nlp/                # NLP models (e.g., TinyBERT)
│
├── notebooks/
│   ├── DataSimulation.ipynb
│   └── Patient_Appointment_Analysis.ipynb
│
├── src/
│   ├── config.py           # Central configuration
│   ├── preprocessing.py    # Data preprocessing
│   ├── supervised.py       # Supervised ML functions
│   ├── unsupervised.py     # Unsupervised ML functions
│   ├── nlp_pipeline.py     # NLP pipeline (TinyBERT, emotion detection)
│   ├── metrics.py          # Evaluation metrics
│   ├── plots.py            # Visualization utilities
│   └── streamlit_app.py    # Streamlit app entry point
│
├── tests/
│   ├── test_nlp_pipeline.py
│   ├── test_preprocessing.py
│   ├── test_supervised.py
│   ├── test_unsupervised.py
│   └── test_tinybert.py
│
└── requirements.txt
```

## Key Features

### NLP Pipeline
- TinyBERT-based sentiment analysis (CPU-optimized)
- Detection of 5 emotions: anxiety, stress, confusion, hopeful, fear
- Smart negation handling and calibration for balanced predictions
- Confidence scoring for sentiment outputs

### Visualization
- Sentiment and emotion distribution analysis
- Emotion-sentiment correlation heatmaps
- Training progress and attendance-based emotion plots

### Configuration Management
- Centralized settings in `config.py`
- Hyperparameters, sentiment keywords, calibration examples
- Device selection (CPU/CUDA) support

## Setup Instructions

1. **Clone the repository:**
  ```bash
  git clone <repository-url>
  ```

2. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

3. **Run the analysis notebook:**
  ```bash
  jupyter notebook notebooks/Patient_Appointment_Analysis.ipynb
  ```