# Patient Appointments Show/No Show Prediction and Analysis

## Overview
This project aims to predict patient appointment outcomes (show/no-show) and analyze various factors influencing these outcomes. It utilizes machine learning techniques, including supervised and unsupervised learning, as well as natural language processing (NLP) for analyzing patient sentiments and reasons for no-shows.

## Project Structure
The project is organized into several directories and files:

- **data/**: Contains raw and processed datasets.
  - **raw/**: Directory for raw data files.
  - **processed/**: Directory for processed data files.
  - **emotion_variations.csv**: CSV file containing variations of emotion terms.
  - **negation_patterns.csv**: CSV file containing negation patterns for NLP.
  
- **models/**: Contains directories for different types of models.
  - **supervised/**: Directory for supervised learning models.
  - **unsupervised/**: Directory for unsupervised learning models.
  - **nlp/**: Directory for NLP models including TinyBERT sentiment analysis models.
  
- **notebooks/**: Jupyter notebooks for various analyses.
  - **DataSimulation.ipynb**: Data simulation for testing.
  - **Patient_Appointment_Analysis.ipynb**: Comprehensive analysis notebook with NLP and visualization.

- **src/**: Source code for the project.
  - **config.py**: Configuration settings including NLP model parameters and hyperparameters.
  - **preprocessing.py**: Data preprocessing functions.
  - **supervised.py**: Supervised learning functions.
  - **unsupervised.py**: Unsupervised learning functions.
  - **nlp_pipeline.py**: Enhanced NLP processing with TinyBERT sentiment analysis and emotion detection.
  - **metrics.py**: Model evaluation functions.
  - **plots.py**: Advanced visualization functions for sentiment and emotion analysis.
  - **streamlit_app.py**: Streamlit application entry point.

- **tests/**: Unit tests for the project modules.
  - **test_nlp_pipeline.py**: Tests for the NLP pipeline including TinyBERT sentiment analysis.
  - **test_preprocessing.py**: Tests for data preprocessing.
  - **test_supervised.py**: Tests for supervised models.
  - **test_unsupervised.py**: Tests for unsupervised models.
  - **test_tinybert.py**: Dedicated test script for TinyBERT sentiment analysis.

- **requirements.txt**: Lists project dependencies with version specifications.

## Key Features

### Enhanced NLP Pipeline
- TinyBERT-based sentiment analysis for efficient performance on CPU
- Detection of 5 emotional states (anxiety, stress, confusion, hopeful, fear)
- Smart negation handling for improved emotion detection
- Calibration mechanism for better sentiment prediction balance
- Confidence scoring for sentiment predictions

### Advanced Visualization
- Sentiment distribution analysis
- Emotion-sentiment correlation heatmaps
- Training progress visualization
- Emotion distribution by appointment attendance

### Configuration Management
- Centralized configuration in config.py
- Model hyperparameters and settings in one location
- Sentiment keywords and calibration examples
- Environment-aware device selection (CPU/CUDA)

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the analysis notebook:
   ```
   jupyter notebook notebooks/PatientAppointmentAnalysis.ipynb
   ```