# Configuration settings for the Patient Appointments Show/No Show Prediction and Analysis project

import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configuration settings for the Patient Appointments Show/No Show Prediction and Analysis project
RUN_CONFIGURATION = [
    { 'step': 'dataload', 'enabled': True },
    { 'step': 'data_preprocess', 'enabled': True },
    { 'step': 'eda', 'enabled': False },
    { 'step': 'supervised_logistic_regression', 'enabled': False },
    { 'step': 'supervised_random_forest', 'enabled': True },
    { 'step': 'unsupervised_pca', 'enabled': True },
    { 'step': 'unsupervised_kmeans', 'enabled': True },
    { 'step': 'unsupervised_gmm', 'enabled': True },
    { 'step': 'nlp_sentiment_analysis', 'enabled': False },
    { 'step': 'nlp_noshow_prediction', 'enabled': True },
    { 'step': 'nlp_topic_modeling', 'enabled': True }
]

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
    EMOTION_VARIATIONS = {
        'anxiety': ['anxious', 'worried', 'nervous', 'uneasy', 'apprehensive', 'tense', 'fearful', 'panic', 'fretful', 'troubled', 'uptight', 'restless', 'agitated', 'concerned', 'distressed', 'anxiousness', 'anxiety disorder', 'anxiously', 'anxiety attack', 'anxiety symptoms', 'high anxiety', 'mild anxiety', 'severe anxiety', 'crippling anxiety', 'constant anxiety', 'worsening anxiety', 'debilitating anxiety', 'overwhelming anxiety', 'recurring anxiety', 'heightened anxiety', 'increasing anxiety', 'anxiety level', 'health anxiety', 'social anxiety', 'anticipatory anxiety', 'generalized anxiety', 'performance anxiety', 'emotional anxiety', 'anxiety feeling', 'anxiety inducing', 'anxiety-related', 'worry', 'feeling anxious'],
        'stress': ['stressed', 'overwhelmed', 'pressure', 'tense', 'strain', 'burden', 'distress', 'burnout', 'hassled', 'taxed', 'exhausted', 'strained', 'hectic', 'harried', 'fatigued', 'stressful', 'stress levels', 'stressfully', 'stressed out', 'burnout syndrome', 'chronic stress', 'acute stress', 'under stress', 'stress reaction', 'mental stress', 'physical stress', 'environmental stress', 'occupational stress', 'daily stress', 'emotional stress', 'work stress', 'stress factor', 'stress management', 'stress response', 'stress symptoms', 'stress levels', 'high stress', 'maximum stress', 'constant stress', 'prolonged stress', 'stress-related', 'strain', 'feeling stressed'],
        'confusion': ['confused', 'perplexed', 'puzzled', 'disoriented', 'uncertain', 'unclear', 'bewildered', 'ambiguous', 'lost', 'disorganized', 'muddled', 'baffled', 'mystified', 'disconcerted', 'unsure', 'confusing', 'confusion state', 'confusedly', 'total confusion', 'mental confusion', 'cognitive confusion', 'momentarily confused', 'thoroughly confused', 'utterly confused', 'somewhat confused', 'increasingly confused', 'slight confusion', 'temporary confusion', 'persistent confusion', 'profound confusion', 'complete confusion', 'confusion about', 'causing confusion', 'added confusion', 'leads to confusion', 'resulting confusion', 'extreme confusion', 'growing confusion', 'general confusion', 'increasing confusion', 'confusion-related', 'misunderstanding', 'feeling confused'],
        'hopeful': ['hope', 'optimistic', 'positive', 'confident', 'encouraged', 'promising', 'anticipation', 'reassured', 'upbeat', 'enthusiastic', 'expectant', 'inspired', 'buoyant', 'cheery', 'motivated', 'hopeful feeling', 'hopefulness', 'hopefully', 'high hopes', 'renewed hope', 'cautiously hopeful', 'feeling hopeful', 'remaining hopeful', 'cautious optimism', 'reasonable hope', 'significant hope', 'unexpected hope', 'genuine hope', 'quiet hope', 'distant hope', 'glimmer of hope', 'spark of hope', 'sign of hope', 'rays of hope', 'beacon of hope', 'message of hope', 'chance of hope', 'foundation of hope', 'element of hope', 'sustained hope', 'hope-filled', 'encouraging', 'feeling hopeful'],
        'fear': ['afraid', 'scared', 'frightened', 'terrified', 'panic', 'dread', 'horror', 'alarmed', 'petrified', 'threatened', 'intimidated', 'worried', 'insecure', 'vulnerable', 'uncomfortable', 'fearfulness', 'fearsome', 'fearfully', 'fear of', 'fight or flight', 'paralyzing fear', 'intense fear', 'fear response', 'deep fear', 'fear of unknown', 'persistent fear', 'gripping fear', 'acute fear', 'mortal fear', 'growing fear', 'deepening fear', 'fear in mind', 'overcome fear', 'face your fear', 'crippling fear', 'nagging fear', 'instilling fear', 'persistent fear', 'perpetual fear', 'intense fear', 'fear-inducing', 'scary', 'feeling afraid']
    }

# Load negation patterns from CSV or fall back to constants
try:
    import pandas as pd
    df_neg = pd.read_csv(NEGATION_PATTERNS_PATH, header=None)
    NEGATION_PATTERNS = df_neg.iloc[:, 0].dropna().tolist()
except Exception:
    NEGATION_PATTERNS = [
        r'not\s+\w+', r'no\s+\w+', r"don't\s+\w+", r"doesn't\s+\w+", r"didn't\s+\w+",
        r"isn't\s+\w+", r"aren't\s+\w+", r"wasn't\s+\w+", r"weren't\s+\w+",
        r"haven't\s+\w+", r"hasn't\s+\w+", r"hadn't\s+\w+", r"won't\s+\w+", r"wouldn't\s+\w+",
        r"can't\s+\w+", r"cannot\s+\w+", r"couldn't\s+\w+", r"shouldn't\s+\w+",
        r'never\s+\w+', r'neither\s+\w+', r'nor\s+\w+', r'none\s+\w+', r'without\s+\w+',
        r'absence\s+of\s+\w+', r'lack\s+of\s+\w+', r'free\s+from\s+\w+', r'deny\s+\w+', r'denies\s+\w+',
        r'refused\s+\w+', r'refute\s+\w+', r'reject\s+\w+', r'nothing\s+\w+', r'nobody\s+\w+',
        r'nowhere\s+\w+', r'rarely\s+\w+', r'seldom\s+\w+', r'hardly\s+\w+', r'barely\s+\w+',
        r'scarcely\s+\w+', r'at\s+all', r'whatsoever', r'lacks', r'free from',
        r'shows no', r'exhibiting no', r'denies', r'denying', r'lacking', r'avoided',
        r'avoiding', r'ruled out', r'no significant', r'minimal', r'no evidence of',
        r'no indication of', r'no signs of', r'absence of any', r'not even', r'shows absence of',
        r'completely absent', r'effectively ruled out', r'is absent', r'are absent',
        r'ruled out for', r'explicitly denied', r'patient denies', r'not bothered by',
        r'not experiencing', r'not feeling', r'not having', r'not at all',
        r'never had', r'rarely if ever', r'infrequently', r'not reported'
    ]

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
    'batch_size': 16,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'epochs': 3,
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

# Hyperparameters for models (example)
HYPERPARAMETERS = {
    'tinybert': [
        {'learning_rate': 5e-5, 'batch_size': 16, 'epochs': 2, 'patience': 1, 'accumulation_steps': 4},
        {'learning_rate': 1e-4, 'batch_size': 16, 'epochs': 2, 'patience': 1, 'accumulation_steps': 4},
    ],
    'tiny_clinicalbert': [
        {
            "model_name": 'nlpie/tiny-clinicalbert',
            "max_length": 128,
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 2e-5
        }
    ]
}

# Model export paths for sentiment analysis
SENTIMENT_MODEL_EXPORT_PATH_RAW = os.path.join(NLP_MODEL_DIR, 'sentiment_analysis_raw')
SENTIMENT_MODEL_EXPORT_PATH_OPTIMIZED = os.path.join(NLP_MODEL_DIR, 'sentiment_analysis_optimized')

PREDICTION_MODEL_EXPORT_PATH = os.path.join(NLP_MODEL_DIR, 'prediction')

def is_step_enabled(step_name):
    return any(step for step in RUN_CONFIGURATION if step['step'] == step_name and step['enabled'] is True)