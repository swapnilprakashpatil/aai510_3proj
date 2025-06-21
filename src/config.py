# Configuration settings for the Patient Appointments Show/No Show Prediction and Analysis project

import os
import sys
import json
from dotenv import load_dotenv

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load .env file
load_dotenv()

# Read each pipeline step from .env separately
RUN_CONFIGURATION = [
    { 'step': 'dataload', 'enabled': os.getenv('RUN_DATALOAD', 'true').lower() == 'true' },
    { 'step': 'data_preprocess', 'enabled': os.getenv('RUN_DATA_PREPROCESS', 'true').lower() == 'true' },
    { 'step': 'eda', 'enabled': os.getenv('RUN_EDA', 'true').lower() == 'true' },
    { 'step': 'supervised_logistic_regression', 'enabled': os.getenv('RUN_SUPERVISED_LOGISTIC_REGRESSION', 'false').lower() == 'true' },
    { 'step': 'supervised_random_forest', 'enabled': os.getenv('RUN_SUPERVISED_RANDOM_FOREST', 'true').lower() == 'true' },
    { 'step': 'unsupervised_pca', 'enabled': os.getenv('RUN_UNSUPERVISED_PCA', 'true').lower() == 'true' },
    { 'step': 'unsupervised_kmeans', 'enabled': os.getenv('RUN_UNSUPERVISED_KMEANS', 'true').lower() == 'true' },
    { 'step': 'unsupervised_gmm', 'enabled': os.getenv('RUN_UNSUPERVISED_GMM', 'true').lower() == 'true' },
    { 'step': 'nlp_sentiment_analysis', 'enabled': os.getenv('RUN_NLP_SENTIMENT_ANALYSIS', 'true').lower() == 'true' },
    { 'step': 'nlp_noshow_prediction', 'enabled': os.getenv('RUN_NLP_NOSHOW_PREDICTION', 'true').lower() == 'true' },
    { 'step': 'nlp_topic_modeling', 'enabled': os.getenv('RUN_NLP_TOPIC_MODELING', 'true').lower() == 'true' },
]

# Configuration settings for the Patient Appointments Show/No Show Prediction and Analysis project
# RUN_CONFIGURATION = json.loads(os.getenv('RUN_CONFIGURATION'))

# Base directory of the project
BASE_DIR = project_root

# Directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
SUPERVISED_MODEL_DIR = os.path.join(MODEL_DIR, 'supervised')
UNSUPERVISED_MODEL_DIR = os.path.join(MODEL_DIR, 'unsupervised')
NLP_MODEL_DIR = os.path.join(MODEL_DIR, 'nlp')
PLOT_DIR = os.path.join(BASE_DIR, 'plots')

# File paths for data simulation
INPUT_PATH = os.path.join(RAW_DATA_DIR, 'dataset_original.csv')
OUTPUT_PATH = os.path.join(DATA_DIR, 'dataset.csv')
DATASET_PATH = os.path.join(DATA_DIR, 'dataset.csv')
EMOTION_VARIATIONS_PATH = os.path.join(DATA_DIR, 'emotion_variations.csv')
NEGATION_PATTERNS_PATH = os.path.join(DATA_DIR, 'negation_patterns.csv')
CLINICAL_TERMS_RULES_PATH = os.path.join(os.path.dirname(__file__), '../data/synthetic_reasons/clinical_terms_rules.csv')

# General configurations
RANDOM_STATE = 42
N_JOBS = -1  # Use all available cores for parallel processing

# Emotion states for NLP pipeline
EMOTION_STATES = ['anxiety', 'stress', 'confusion', 'hopeful', 'fear']

# Load emotion variations from CSV or fall back to constants
try:
    import pandas as pd
    df_var = pd.read_csv(EMOTION_VARIATIONS_PATH)
    EMOTION_VARIATIONS = df_var.groupby('emotion')['variation'].apply(list).to_dict()
    # Ensure all emotion states are in the dictionary
    for emotion in EMOTION_STATES:
        if emotion not in EMOTION_VARIATIONS:
            EMOTION_VARIATIONS[emotion] = [emotion]
except Exception as e:
    print(f"Error loading emotion variations: {e}")

# Load negation patterns from CSV or fall back to constants
try:
    import pandas as pd
    df_neg = pd.read_csv(NEGATION_PATTERNS_PATH, header=None)
    NEGATION_PATTERNS = df_neg.iloc[:, 0].dropna().tolist()
except Exception as e:
    print(f"Error loading negation patterns: {e}")

# Model names for centralized reference
MODEL_NAMES = {
    'TINY': 'prajjwal1/bert-tiny',
    'LARGE': 'bert-base-uncased',
    'DEFAULT': 'prajjwal1/bert-tiny'  # Default model to use
}

# NLP model configuration
NLP_CONFIG = {
    'default_model': MODEL_NAMES['DEFAULT'],
    'large_model': MODEL_NAMES['LARGE'],
    'tiny_model': MODEL_NAMES['TINY'],
    'device': 'cuda' if os.environ.get('USE_CUDA', 'False').lower() == 'true' else 'cpu',
    'max_length': 128,
    'learning_rate': 3e-5,  # Lowered for more stable training
    'batch_size': 8,        # Smaller batch size for better generalization
    'epochs': 6,            # Increased epochs for more training
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'eval_steps': 10,  # now configurable
    'save_steps': 10,  # now configurable
    'logging_steps': 10,  # now configurable
    'sentiment_threshold': 0.55,  # Threshold for positive sentiment prediction
    'temperature': 1.2,  # Temperature for softening predictions
    'num_labels': 2,  # Binary sentiment: negative/positive
    'label_mapping': {0: 'negative', 1: 'positive'},
    'id_label_mapping': {0: 'negative', 1: 'positive'},
    'label_id_mapping': {'negative': 0, 'positive': 1}
}

# Hyperparameters for models (expanded)
HYPERPARAMETERS = {
    'tinybert': [
        {'learning_rate': 5e-5, 'batch_size': 16, 'epochs': 2, 'patience': 1, 'accumulation_steps': 4},
        {'learning_rate': 1e-4, 'batch_size': 16, 'epochs': 2, 'patience': 1, 'accumulation_steps': 4},
        {'learning_rate': 3e-5, 'batch_size': 8, 'epochs': 6, 'patience': 2, 'accumulation_steps': 2},
        {'learning_rate': 2e-5, 'batch_size': 32, 'epochs': 4, 'patience': 2, 'accumulation_steps': 4},
        {'learning_rate': 1e-5, 'batch_size': 16, 'epochs': 8, 'patience': 3, 'accumulation_steps': 2},
    ],
    'tiny_clinicalbert': [
        {
            "model_name": 'nlpie/tiny-clinicalbert',
            "max_length": 128,
            "epochs": 6,
            "batch_size": 8,
            "learning_rate": 3e-5
        },
        {
            "model_name": 'nlpie/tiny-clinicalbert',
            "max_length": 256,
            "epochs": 8,
            "batch_size": 16,
            "learning_rate": 2e-5
        }
    ]
}

# Model export paths for sentiment analysis
SENTIMENT_MODEL_EXPORT_PATH_RAW = os.path.join(NLP_MODEL_DIR, 'sentiment_analysis_raw')
SENTIMENT_MODEL_EXPORT_PATH_OPTIMIZED = os.path.join(NLP_MODEL_DIR, 'sentiment_analysis_optimized')
# Model export path for  prediction
PREDICTION_MODEL_EXPORT_PATH = os.path.join(NLP_MODEL_DIR, 'prediction')
# Model export path for topic modeling
TOPIC_MODEL_EXPORT_PATH = os.path.join(NLP_MODEL_DIR, 'topic_model')

def is_step_enabled(step_name):
    return any(step for step in RUN_CONFIGURATION if step['step'] == step_name and step['enabled'] is True)