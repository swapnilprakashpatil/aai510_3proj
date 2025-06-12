import pandas as pd
import numpy as np
from src.config import EMOTION_STATES, NLP_CONFIG

class EmotionPostProcessor:
    def __init__(self, emotion_variations_path=None, negation_patterns_path=None, thresholds=None):
        self.emotional_states = EMOTION_STATES

        # Use thresholds from config or override if provided
        default_thresholds = NLP_CONFIG.get('emotion_thresholds', {
            'anxiety': 0.4,
            'stress': 0.7,
            'confusion': 0.4,
            'hopeful': 0.5,
            'fear': 0.7
        })
        self.thresholds = default_thresholds.copy()
        if thresholds:
            self.thresholds.update(thresholds)

        # Use default paths from config if not provided
        if not emotion_variations_path:
            emotion_variations_path = NLP_CONFIG.get('emotion_variations_path', None)
        if not negation_patterns_path:
            negation_patterns_path = NLP_CONFIG.get('negation_patterns_path', None)

        # Load emotion variations from CSV
        self.word_variations = self._load_emotion_variations(emotion_variations_path)
        # Load negation patterns from CSV
        self.negation_phrases = self._load_negation_patterns(negation_patterns_path)
    
    def _load_emotion_variations(self, csv_path):
        if not csv_path:
            raise ValueError("Emotion variations CSV path must be provided")
        df = pd.read_csv(csv_path)
        variations = {}
        for emotion in df['emotion'].unique():
            variations[emotion] = df[df['emotion'] == emotion]['variation'].tolist()
        return variations
    
    def _load_negation_patterns(self, csv_path):
        if not csv_path:
            raise ValueError("Negation patterns CSV path must be provided")
        df = pd.read_csv(csv_path)
        return df['negation_phrase'].tolist()
    
    def _detect_manual_emotions(self, text):
        manual_detections = {}
        text_lower = text.lower()
        for emotion, variations in self.word_variations.items():
            for variant in variations:
                if variant in text_lower and not any(neg + variant in text_lower for neg in [' no ', ' not ', 'without ', 'absence of ']):
                    manual_detections[emotion] = True
                    break
        return manual_detections
    
    def _detect_negated_emotions(self, text):
        negated_emotions = []
        text_lower = text.lower()
        if "no significant anxiety, stress, or fear" in text_lower:
            negated_emotions.extend(['anxiety', 'stress', 'fear'])
        for emotion in self.emotional_states:
            for variant in self.word_variations.get(emotion, []):
                if f"no {variant}" in text_lower:
                    negated_emotions.append(emotion)
                    break
                if any(f"{phrase}{variant}" in text_lower for phrase in self.negation_phrases):
                    negated_emotions.append(emotion)
                    break
                if f"no significant {variant}" in text_lower:
                    negated_emotions.append(emotion)
                    break
        if "no significant" in text_lower and "related to health conditions" in text_lower:
            for emotion in ['anxiety', 'stress', 'fear']:
                if emotion not in negated_emotions and emotion in self.emotional_states:
                    negated_emotions.append(emotion)
        return negated_emotions
    
    def predict(self, text, model, tokenizer, device):
        model.eval()
        manual_detections = self._detect_manual_emotions(text)
        negated_emotions = self._detect_negated_emotions(text)
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=64,  # Using smaller max length for prediction
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        with np.errstate(over='ignore'):
            import torch
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.sigmoid(logits).cpu().numpy()[0]
        results = {}
        for i, emotion in enumerate(self.emotional_states):
            threshold = self.thresholds.get(emotion, 0.5)
            if emotion in negated_emotions:
                results[emotion] = False
            elif emotion in manual_detections:
                results[emotion] = True
            else:
                results[emotion] = bool(probs[i] >= threshold)
        return results
