import pytest
import numpy as np
import torch
import os
from src.sentiment_analysis import SentimentAnalysisModel
from src.config import EMOTION_STATES, NLP_CONFIG, SENTIMENT_MODEL_EXPORT_PATH_RAW, SENTIMENT_MODEL_EXPORT_PATH_OPTIMIZED
from src.emotion_postprocessor import EmotionPostProcessor

# Example test texts
TEST_TEXTS = [
    "Patient is hopeful and shows no significant anxiety, stress, or fear related to health conditions.",
    "Patient expresses fear and anxiety about high blood pressure and possible complications.",
    "Elderly patient expresses fear of declining health, confusion about medications, and stress related to mobility issues.",
    "Patient (minor) is anxious and fearful about medical procedures, sometimes confused by instructions, and stressed by separation from family.",
    "Patient is calm and shows no signs of stress, anxiety, or fear during the appointment.",
    "Patient is confused about the medication schedule and expresses frustration.",
    "Patient is hopeful about recovery but still experiences occasional stress.",
    "Patient is fearful of surgery and anxious about the outcome.",
    "Patient expresses both hope and anxiety regarding the new treatment plan.",
    "Patient is neither anxious nor fearful, but is confused by the instructions."
]

EXPECTED_EMOTIONS = [
    ["hopeful"],
    ["fear", "anxiety"],
    ["fear", "confusion", "stress"],
    ["anxiety", "fear", "confusion", "stress"],
    [],  # No negative emotions expected
    ["confusion"],
    ["hopeful", "stress"],
    ["fear", "anxiety"],
    ["hopeful", "anxiety"],
    ["confusion"]
]

def _load_model_and_tokenizer(export_path):
    model_dir = export_path
    if os.path.exists(model_dir):
        model = SentimentAnalysisModel.load_from_pretrained(model_dir, device=NLP_CONFIG['device'])
        tokenizer = model.tokenizer
        return model, tokenizer, model_dir
    else:
        pytest.skip(f"Trained model not found at {model_dir}. Run hyperparameter tuning and export the best model first.")

def _print_and_score_results(results, test_texts, expected_emotions):
    total = len(test_texts)
    passed = 0
    for idx, (text, pred) in enumerate(zip(test_texts, results)):
        pred_dict = {emo: int(val) for emo, val in zip(EMOTION_STATES, pred)}
        test_passed = True
        for emo in expected_emotions[idx]:
            if pred_dict[emo] != 1:
                test_passed = False
                print(f"\u274C Test FAILED for: {text}\nPrediction: {pred_dict}\nExpected: {expected_emotions[idx]}\n")
                assert False, f"Expected emotion '{emo}' to be present in: {text} (got {pred_dict})"
        if test_passed:
            passed += 1
            print(f"\u2705 Test PASSED for: {text}\nPrediction: {pred_dict}\nExpected: {expected_emotions[idx]}\n")
    return passed, total

def test_sentiment_model_predictions_raw():
    print("\n--- Running test_sentiment_model_predictions_raw ---")
    model, tokenizer, _ = _load_model_and_tokenizer(SENTIMENT_MODEL_EXPORT_PATH_RAW)
    model.model.eval()
    results = []
    for text in TEST_TEXTS:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=NLP_CONFIG['max_length'],
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(NLP_CONFIG['device'])
        attention_mask = encoding['attention_mask'].to(NLP_CONFIG['device'])
        with torch.no_grad():
            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            preds = (probs >= 0.5).astype(int)
        results.append(preds)
    for pred in results:
        assert len(pred) == len(EMOTION_STATES)
        assert all((p == 0 or p == 1) for p in pred)
    passed, total = _print_and_score_results(results, TEST_TEXTS, EXPECTED_EMOTIONS)
    print(f"\nTest score (raw): {passed}/{total} passed.")
    # Force output to always show in pytest summary
    assert True

def test_sentiment_model_predictions_optimized(capfd):
    print("\n--- Running test_sentiment_model_predictions_optimized ---")
    model, tokenizer, model_dir = _load_model_and_tokenizer(SENTIMENT_MODEL_EXPORT_PATH_OPTIMIZED)
    post_processor = EmotionPostProcessor(
        emotion_variations_path=os.path.join(model_dir, 'emotion_variations.csv'),
        negation_patterns_path=os.path.join(model_dir, 'negation_patterns.csv')
    )
    model.model.eval()
    results = []
    for text in TEST_TEXTS:
        emotion_results = post_processor.predict(text, model.model, tokenizer, NLP_CONFIG['device'])
        preds = [int(emotion_results[emo]) for emo in EMOTION_STATES]
        results.append(preds)
    for pred in results:
        assert len(pred) == len(EMOTION_STATES)
        assert all((p == 0 or p == 1) for p in pred)
    passed, total = _print_and_score_results(results, TEST_TEXTS, EXPECTED_EMOTIONS)
    print(f"\nTest score (optimized): {passed}/{total} passed.")
    # Force output to always show in pytest summary
    out, _ = capfd.readouterr()
    print(out)
    assert True
