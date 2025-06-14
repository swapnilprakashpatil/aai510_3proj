import os
import pytest
from src.clinical_notes_prediction import ClinicalNotesNoShowPredictor
from src.config import PREDICTION_MODEL_EXPORT_PATH

def test_exported_clinicalbert_model():
    model_dir = PREDICTION_MODEL_EXPORT_PATH
    assert os.path.exists(model_dir), f"Exported model directory not found: {model_dir}"
    loaded_predictor = ClinicalNotesNoShowPredictor()
    loaded_predictor.model = loaded_predictor.model.from_pretrained(model_dir)
    loaded_predictor.tokenizer = loaded_predictor.tokenizer.from_pretrained(model_dir)
    samples = [
        "Patient is anxious and stressed about the appointment.",
        "Patient is hopeful and looking forward to the visit.",
        "Patient missed previous appointments due to transportation issues.",
        "Patient is calm and has attended all previous appointments.",
        "Patient is worried about the outcome and has a history of no-shows.",
        "Patient is excited to see the doctor and has no prior no-shows.",
        "Patient is frustrated due to long wait times and missed last appointment.",
        "Patient is positive and always attends appointments.",
        "Patient is nervous and did not show up last time.",
        "Patient is relaxed and has a good attendance record."
    ]
    expected = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # Adjust as appropriate for your model
    preds = loaded_predictor.predict(samples)
    passed = 0
    failed = 0
    for i, (text, pred, exp) in enumerate(zip(samples, preds, expected)):
        print(f"Sample {i+1}: {text}")
        print(f"  Prediction: {pred}, Expected: {exp}")
        if pred == exp:
            print("  \u2705 Prediction matches expected!")
            passed += 1
        else:
            print("  \u274C Prediction does not match expected.")
            failed += 1
    print(f"\nTest summary: {passed} passed, {failed} failed out of {len(samples)}.")
    assert passed == len(samples), f"Some ClinicalNotesNoShowPredictor model tests failed: {failed} failed."
