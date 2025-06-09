from typing import List, Tuple, Dict, Union, Optional
import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
import logging
import json

# Import config
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
# Additional NLTK resources needed for text processing
nltk_resources = ['punkt', 'stopwords']
for resource in nltk_resources:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)

# Constants for emotion detection - import from config
EMOTION_STATES = config.EMOTION_STATES
EMOTION_VARIATIONS = config.EMOTION_VARIATIONS
NEGATION_PATTERNS = config.NEGATION_PATTERNS

class TextDataset(Dataset):
    """Dataset class for text classification with TinyBERT"""
    
    def __init__(self, texts, labels=None, tokenizer=None, max_length=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        # Use max_length from config if not specified
        self.max_length = max_length if max_length else config.NLP_CONFIG['max_length']

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Remove batch dimension added by tokenizer when return_tensors='pt'
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # Add label if available
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return item

class NLPipeline:
    """Enhanced NLP Pipeline with TinyBERT and emotion detection"""
    
    def __init__(self, model_name=None, device=None):
        """
        Initialize the NLP Pipeline
        
        Args:
            model_name (str): Name of the pre-trained model to use
            device (str): Device to use for computation ('cpu' or 'cuda')
        """
        # Use model name from config if not specified
        self.model_name = model_name if model_name else config.NLP_CONFIG['default_model']
        
        # Use device from config if not specified
        self.device = device if device else config.NLP_CONFIG['device']
        logger.info(f"Using device: {self.device}")
        
        # TF-IDF components
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced'
        )
        
        # TinyBERT components
        self.tokenizer = None
        self.model = None
        self.load_bert_model()  # Initialize BERT model by default
        
        # Metrics storage
        self.metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'confusion_matrix': None
        }
        
        # Best hyperparameters
        self.best_params = None
        
        # BERT components
        self.tokenizer = None
        self.model = None
        self.emotion_classifier = None
        
        # Metrics storage
        self.metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'confusion_matrix': None
        }
        
        # Best hyperparameters
        self.best_params = {}
        
    def load_bert_model(self, num_labels=None):
        """
        Load TinyBERT model for sentiment analysis
        
        Args:
            num_labels (int): Number of sentiment classes (default: from config)
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            # Use number of labels from config if not specified
            if num_labels is None:
                num_labels = config.NLP_CONFIG['num_labels']
            
            logger.info(f"Loading model {self.model_name} with {num_labels} labels...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=num_labels
            )
            self.model.to(self.device)
            logger.info(f"Model loaded successfully on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def train(self, 
              train_texts: List[str], 
              train_labels: List[int],
              val_texts: Optional[List[str]] = None,
              val_labels: Optional[List[int]] = None,
              batch_size: Optional[int] = None,
              learning_rate: Optional[float] = None,
              epochs: Optional[int] = None,
              warmup_ratio: Optional[float] = None):
        """
        Train the TinyBERT model for sentiment analysis
        
        Args:
            train_texts (List[str]): List of training text samples
            train_labels (List[int]): List of training labels
            val_texts (List[str], optional): List of validation text samples
            val_labels (List[int], optional): List of validation labels
            batch_size (int, optional): Batch size for training
            learning_rate (float, optional): Learning rate for training
            epochs (int, optional): Number of training epochs
            warmup_ratio (float, optional): Ratio of warmup steps
            
        Returns:
            dict: Training metrics
        """
        # Use parameters from config if not specified
        batch_size = batch_size if batch_size else config.NLP_CONFIG['batch_size']
        learning_rate = learning_rate if learning_rate else config.NLP_CONFIG['learning_rate']
        epochs = epochs if epochs else config.NLP_CONFIG['epochs']
        warmup_ratio = warmup_ratio if warmup_ratio else config.NLP_CONFIG['warmup_ratio']
        
        # Create train dataset
        train_dataset = TextDataset(
            texts=train_texts,
            labels=train_labels,
            tokenizer=self.tokenizer
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Create validation dataset if provided
        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_dataset = TextDataset(
                texts=val_texts,
                labels=val_labels,
                tokenizer=self.tokenizer
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size
            )
        
        # Set up optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=config.NLP_CONFIG['weight_decay']
        )
        
        # Calculate total steps
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        # Create scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        logger.info(f"Starting training with {len(train_texts)} samples for {epochs} epochs...")
        self.model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch in train_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Update statistics
                running_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Calculate epoch statistics
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f}")
            
            # Validation if available
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                logger.info(f"Validation - Loss: {val_metrics['loss']:.4f} - Acc: {val_metrics['accuracy']:.4f}")
                
        logger.info("Training completed")
        
        # Final evaluation
        if val_loader:
            final_metrics = self.evaluate(val_loader)
            self.metrics['accuracy'].append(final_metrics['accuracy'])
            self.metrics['precision'].append(final_metrics['precision'])
            self.metrics['recall'].append(final_metrics['recall'])
            self.metrics['f1'].append(final_metrics['f1'])
            self.metrics['confusion_matrix'] = final_metrics['confusion_matrix']
            
            return final_metrics
        
        return {'accuracy': epoch_acc, 'loss': epoch_loss}
    
    def evaluate(self, val_loader=None, test_texts=None, test_labels=None):
        """
        Evaluate the TinyBERT model on validation or test data
        
        Args:
            val_loader (DataLoader, optional): Validation data loader
            test_texts (List[str], optional): List of test text samples
            test_labels (List[int], optional): List of test labels
            
        Returns:
            dict: Evaluation metrics
        """
        # Create test loader if texts and labels are provided
        if val_loader is None and test_texts is not None and test_labels is not None:
            test_dataset = TextDataset(
                texts=test_texts,
                labels=test_labels,
                tokenizer=self.tokenizer
            )
            val_loader = DataLoader(
                test_dataset,
                batch_size=config.NLP_CONFIG['batch_size']
            )
        
        if val_loader is None:
            logger.error("No validation data provided")
            return None
        
        logger.info("Starting evaluation...")
        self.model.eval()
        
        # Initialize metrics
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # Update statistics
                running_loss += loss.item()
                
                # Get predictions
                _, predicted = torch.max(logits, 1)
                
                # Store predictions and labels for metrics
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        cm = confusion_matrix(all_labels, all_preds)
        
        # Calculate average loss
        avg_loss = running_loss / len(val_loader)
        
        logger.info(f"Evaluation - Acc: {accuracy:.4f} - F1: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': avg_loss,
            'confusion_matrix': cm
        }
    
    def predict(self, texts, raw_outputs=False, temperature=None):
        """
        Predict sentiment of texts using TinyBERT
        
        Args:
            texts (List[str] or str): Text(s) to predict sentiment for
            raw_outputs (bool): Whether to return raw model outputs
            temperature (float): Temperature for softening predictions
            
        Returns:
            List or np.ndarray: Predictions
        """
        # Normalize input to list
        if isinstance(texts, str):
            texts = [texts]
        
        # Use temperature from config if not specified
        temperature = temperature if temperature else config.NLP_CONFIG['temperature']
        
        # Create dataset
        dataset = TextDataset(
            texts=texts,
            tokenizer=self.tokenizer
        )
        dataloader = DataLoader(
            dataset,
            batch_size=config.NLP_CONFIG['batch_size']
        )
        
        logger.info(f"Predicting sentiment for {len(texts)} texts...")
        self.model.eval()
        
        all_outputs = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                
                # Apply temperature to soften predictions if needed
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(logits, dim=1)
                
                # Store outputs
                all_outputs.extend(probs.cpu().numpy())
        
        all_outputs = np.array(all_outputs)
        
        # Return raw outputs if requested
        if raw_outputs:
            return all_outputs
        
        # Otherwise, return predicted classes based on threshold
        threshold = config.NLP_CONFIG['sentiment_threshold']
        predictions = []
        
        for output in all_outputs:
            # For binary sentiment (0=negative, 1=positive)
            # Convert to label using threshold
            label = 1 if output[1] >= threshold else 0
            predictions.append(label)
        
        return predictions
    
    def predict_with_explanation(self, texts):
        """
        Predict sentiment with explanation
        
        Args:
            texts (List[str] or str): Text(s) to predict sentiment for
            
        Returns:
            List[Dict]: Predictions with explanations
        """
        # Normalize input to list
        if isinstance(texts, str):
            texts = [texts]
        
        # Get raw predictions
        raw_outputs = self.predict(texts, raw_outputs=True)
        
        # Convert to sentiment labels and confidences
        results = []
        
        for i, output in enumerate(raw_outputs):
            # Get sentiment label
            sentiment_id = 1 if output[1] >= config.NLP_CONFIG['sentiment_threshold'] else 0
            sentiment_label = config.NLP_CONFIG['label_mapping'][sentiment_id]
            confidence = output[sentiment_id]
            
            # Detect emotion terms
            emotions_detected = self.detect_emotions(texts[i])
            
            # Detect negation
            contains_negation = self.detect_negation(texts[i])
            
            # Create result dictionary
            result = {
                'text': texts[i],
                'sentiment': sentiment_label,
                'confidence': float(confidence),
                'emotions_detected': emotions_detected,
                'contains_negation': contains_negation
            }
            
            results.append(result)
        
        return results
    
    def detect_emotions(self, text):
        """
        Detect emotions in text based on emotion variations
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, List]: Detected emotion terms by emotion
        """
        detected_emotions = {}
        
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Check for each emotion and its variations
        for emotion, variations in EMOTION_VARIATIONS.items():
            found_terms = []
            
            for term in variations:
                # Use word boundary to avoid partial matches
                pattern = r'\b' + re.escape(term.lower()) + r'\b'
                matches = re.findall(pattern, text_lower)
                
                if matches:
                    found_terms.extend(matches)
            
            if found_terms:
                detected_emotions[emotion] = found_terms
        
        return detected_emotions
    
    def detect_negation(self, text):
        """
        Detect negation patterns in text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            bool: Whether text contains negation
        """
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Check for negation patterns
        for pattern in NEGATION_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def save_model(self, path=None):
        """
        Save the TinyBERT model, tokenizer, and metrics
        
        Args:
            path (str): Path to save the model
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Use default path if not specified
            if path is None:
                path = os.path.join(config.NLP_MODEL_DIR, 'tinybert_sentiment')
            
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Save model and tokenizer
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            
            # Save metrics
            metrics_path = os.path.join(path, 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f)
            
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_saved_model(self, path=None):
        """
        Load a saved TinyBERT model, tokenizer, and metrics
        
        Args:
            path (str): Path to load the model from
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            # Use default path if not specified
            if path is None:
                path = os.path.join(config.NLP_MODEL_DIR, 'tinybert_sentiment')
            
            # Load model and tokenizer
            self.model = AutoModelForSequenceClassification.from_pretrained(path)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            
            # Move model to device
            self.model.to(self.device)
            
            # Load metrics if available
            metrics_path = os.path.join(path, 'metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f)
            
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def hyperparameter_tuning(self, train_texts, train_labels, val_texts, val_labels):
        """
        Perform hyperparameter tuning for the TinyBERT model
        
        Args:
            train_texts (List[str]): Training text samples
            train_labels (List[int]): Training labels
            val_texts (List[str]): Validation text samples
            val_labels (List[int]): Validation labels
            
        Returns:
            dict: Best hyperparameters
        """
        logger.info("Starting hyperparameter tuning...")
        
        # Define hyperparameter grid from config
        param_grid = config.HYPERPARAMETERS['tinybert']
        
        best_accuracy = 0
        best_params = {}
        
        # Cartesian product of hyperparameters
        from itertools import product
        param_combinations = list(product(
            param_grid['learning_rate'],
            param_grid['batch_size'],
            param_grid['epochs']
        ))
        
        for lr, bs, epochs in param_combinations:
            logger.info(f"Trying: lr={lr}, batch_size={bs}, epochs={epochs}")
            
            # Reset model for each trial
            self.load_bert_model()
            
            # Train with current hyperparameters
            metrics = self.train(
                train_texts=train_texts,
                train_labels=train_labels,
                val_texts=val_texts,
                val_labels=val_labels,
                learning_rate=lr,
                batch_size=bs,
                epochs=epochs
            )
            
            # Track best parameters
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_params = {
                    'learning_rate': lr,
                    'batch_size': bs,
                    'epochs': epochs
                }
                
                logger.info(f"New best: {best_params} with accuracy {best_accuracy:.4f}")
        
        logger.info(f"Best hyperparameters: {best_params}")
        self.best_params = best_params
        
        # Reset model with best parameters
        self.load_bert_model()
        
        return best_params
    
    def explain_prediction(self, text, prediction):
        """
        Explain a sentiment prediction
        
        Args:
            text (str): Input text
            prediction (int): Predicted sentiment (0=negative, 1=positive)
            
        Returns:
            str: Explanation of the prediction
        """
        # Get sentiment label
        sentiment = config.NLP_CONFIG['label_mapping'][prediction]
        
        # Detect emotions
        emotions = self.detect_emotions(text)
        emotion_explanation = ""
        if emotions:
            emotion_terms = []
            for emotion, terms in emotions.items():
                emotion_terms.append(f"{emotion} ({', '.join(terms)})")
            emotion_explanation = f"Detected emotions: {', '.join(emotion_terms)}. "
        
        # Detect negation
        negation = self.detect_negation(text)
        negation_explanation = "Contains negation which may affect sentiment. " if negation else ""
        
        # Identify keywords
        positive_keywords = []
        negative_keywords = []
        
        for keyword in config.SENTIMENT_KEYWORDS['positive']:
            if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text.lower()):
                positive_keywords.append(keyword)
        
        for keyword in config.SENTIMENT_KEYWORDS['negative']:
            if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text.lower()):
                negative_keywords.append(keyword)
        
        keyword_explanation = ""
        if positive_keywords:
            keyword_explanation += f"Positive keywords: {', '.join(positive_keywords)}. "
        if negative_keywords:
            keyword_explanation += f"Negative keywords: {', '.join(negative_keywords)}. "
        
        # Create explanation
        explanation = f"Predicted sentiment: {sentiment}. "
        explanation += emotion_explanation
        explanation += negation_explanation
        explanation += keyword_explanation
        
        return explanation.strip()
    
    def analyse_sentiment_distribution(self, texts):
        """
        Analyze sentiment distribution in a corpus
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            dict: Sentiment distribution statistics
        """
        # Predict sentiment for all texts
        predictions = self.predict(texts)
        raw_outputs = self.predict(texts, raw_outputs=True)
        
        # Count sentiment labels
        sentiment_counts = {
            'positive': predictions.count(1),
            'negative': predictions.count(0)
        }
        
        # Calculate confidence statistics
        confidences = []
        for i, pred in enumerate(predictions):
            confidence = raw_outputs[i][pred]
            confidences.append(confidence)
        
        # Calculate emotion statistics
        emotion_counts = {emotion: 0 for emotion in EMOTION_STATES}
        
        for text in texts:
            emotions = self.detect_emotions(text)
            for emotion in emotions:
                emotion_counts[emotion] += 1
        
        # Calculate negation statistics
        negation_count = sum(1 for text in texts if self.detect_negation(text))
        
        return {
            'total_texts': len(texts),
            'sentiment_counts': sentiment_counts,
            'sentiment_percentage': {
                'positive': sentiment_counts['positive'] / len(texts) * 100,
                'negative': sentiment_counts['negative'] / len(texts) * 100
            },
            'confidence_stats': {
                'mean': np.mean(confidences),
                'median': np.median(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            'emotion_counts': emotion_counts,
            'emotion_percentage': {
                emotion: count / len(texts) * 100
                for emotion, count in emotion_counts.items()
            },
            'negation_count': negation_count,
            'negation_percentage': negation_count / len(texts) * 100
        }
