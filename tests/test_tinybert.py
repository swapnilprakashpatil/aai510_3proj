import sys
import os
import pandas as pd
import torch

# Add parent directory to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.nlp_sentimentanalysis import NLPipeline, EMOTION_STATES
from src import config

def test_tinybert_sentiment():
    # Initialize NLP pipeline with tiny model for quick testing
    nlp = NLPipeline(model_name=config.MODEL_NAMES['TINY'])
    success = nlp.load_bert_model()
    
    if success:
        print("Model loaded successfully!")
    else:
        print("Error loading model!")
        return

    # Initialize with examples to improve calibration
    nlp.initialize_with_examples()
    
    # Sample texts with expected sentiment
    positive_texts = [
        "I'm looking forward to my appointment next week",
        "The doctor was very helpful and kind",
        "I'm feeling much better after the treatment"
    ]
    
    negative_texts = [
        "I'm worried about my test results",
        "The waiting time was too long and frustrating",
        "I'm not sure if the medication is working"
    ]
    
    # Test positive samples individually
    positive_results = nlp.predict_sentiment(positive_texts)
    print("\nPositive texts test:")
    for i, (text, result) in enumerate(zip(positive_texts, positive_results)):
        print(f"Text: '{text[:30]}...' -> Sentiment: {'positive' if result == 1 else 'negative'}")
    
    # Test negative samples individually
    negative_results = nlp.predict_sentiment(negative_texts)
    print("\nNegative texts test:")
    for i, (text, result) in enumerate(zip(negative_texts, negative_results)):
        print(f"Text: '{text[:30]}...' -> Sentiment: {'positive' if result == 1 else 'negative'}")
    
    # Use the analyze_patient_sentiment method
    all_texts = positive_texts + negative_texts
    results = nlp.analyze_patient_sentiment(all_texts)
    
    print("\nFull analysis results:")
    for i, row in results.iterrows():
        print(f"Text: '{row['text'][:30]}...'")
        print(f"  Sentiment: {row['sentiment_label']}")
        print(f"  Dominant emotion: {row['dominant_emotion']}")
        print(f"  Confidence: {row.get('sentiment_confidence', 'N/A')}")
        print()
    
    # Check counts of positive vs negative predictions
    positive_count = results['sentiment'].sum()
    negative_count = len(results) - positive_count
    
    print(f"Summary:")
    print(f"  Positive predictions: {positive_count}")
    print(f"  Negative predictions: {negative_count}")
    print(f"  Balance (positive/negative): {positive_count/negative_count if negative_count > 0 else 'inf'}")
    
    return results

if __name__ == "__main__":
    test_tinybert_sentiment()
