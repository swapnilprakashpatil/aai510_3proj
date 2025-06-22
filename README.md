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
│   ├── raw/                    # Raw data files
│   ├── synthetic_reasons/      # Synthetic no-show reason templates
│   │   ├── age_based_reasons.csv
│   │   ├── clinical_terms_rules.csv
│   │   ├── gender_based_reasons.csv
│   │   ├── no_show_reasons.csv
│   │   ├── patient_notes_templates.csv
│   │   ├── patient_sentiment_templates.csv
│   │   ├── positive_attendance_reasons.csv
│   │   ├── rule_engine.csv
│   │   └── sms_based_reasons.csv
│   ├── emotion_variations.csv
│   ├── negation_patterns.csv
│   ├── default_stopwords.csv
│   └── dataset.csv             # Main dataset
│
├── models/
│   ├── supervised/             # Supervised learning models
│   ├── unsupervised/           # Unsupervised learning models
│   └── nlp/
│       ├── prediction/
│       │   ├── sentiment_analysis_optimized/   # Optimized sentiment models
│       │   └── sentiment_analysis_raw/         # Raw sentiment models
│       └── topic_model/                        # Topic modeling artifacts
│
├── notebooks/
│   ├── Clustering.ipynb
│   ├── DataSimulation.ipynb
│   ├── EDA.ipynb
│   ├── Final Project Section4-Team 3.ipynb
│   ├── NoShowPrediction.ipynb
│   ├── PatientSentimentAnalysis.ipynb
│   ├── Supervised.ipynb
│   ├── Supervised_Clean.ipynb
│   └── TopicModeling.ipynb
│
├── src/
│   ├── __init__.py
│   ├── clinical_notes_prediction.py            # Clinical note no-show prediction logic
│   ├── clinical_topic_model.py                 # Clinical topic modeling logic
│   ├── clustering.py                           # Clustering utilities
│   ├── config.py                               # Central configuration (hyperparameters, paths, etc.)
│   ├── data_simulator.py                       # Data simulation utilities
│   ├── emotion_postprocessor.py                # Postprocessing for emotion detection
│   ├── helpers.py                              # Helper functions
│   ├── metrics.py                              # Evaluation metrics
│   ├── no_show_prediction.py                   # Modular no-show ML pipeline
│   ├── plots.py                                # Visualization utilities
│   ├── preprocessor.py                         # Data preprocessing
│   ├── sentiment_analysis.py                   # Sentiment analysis logic
│   └── __pycache__/
│
├── tests/
│   ├── __init__.py
│   ├── test_clinical_notes_prediction.py
│   ├── test_sentiment_anlaysis.py
│   ├── test_sentiment_prediction.py
│   └── ...
│
├── app.py                                      # Streamlit app entry point
├── requirements.txt
├── setup.py
└── README.md
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
  git clone https://github.com/swapnilprakashpatil/aai510_3proj.git
  ```

2. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

3. **Run the analysis notebook:**
  ```bash
  jupyter notebook notebooks/Final Project Section4-Team 3.ipynb
  ```