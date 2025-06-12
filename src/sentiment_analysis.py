import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import warnings
import time as time_module  # Renamed to prevent variable shadowing
import psutil
import threading
import itertools
import os
import platform
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from transformers import get_linear_schedule_with_warmup, AutoConfig
from torch.optim import AdamW
from tqdm import tqdm
from src.config import RANDOM_STATE, EMOTION_STATES, MODEL_NAMES, NLP_CONFIG, HYPERPARAMETERS
from src.emotion_postprocessor import EmotionPostProcessor

warnings.filterwarnings("ignore")

class EmotionDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len=128):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        target = self.targets[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.FloatTensor(target)
        }
    
class SentimentAnalysisModel:
    
    def __init__(self, df, emotional_states=EMOTION_STATES, device=NLP_CONFIG['device'], seed=RANDOM_STATE):
        self.df = df
        self.emotional_states = emotional_states
        self.device = device
        self.seed = seed
        self.model_name = NLP_CONFIG['default_model']
        # Use TinyBERT notebook defaults for speed
        self.max_seq_length = 32  # Reduced from 128 for CPU speed
        self.batch_size = 16      # TinyBERT default
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._prepare_data()
        self._init_model()

    def _prepare_data(self):
        self.df['emotion_count'] = self.df[self.emotional_states].sum(axis=1)
        X = self.df['PatientSentiment'].values
        y = self.df[self.emotional_states].values
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed, stratify=self.df['emotion_count'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.15, random_state=self.seed)
        # Use num_workers=0 for CPU, no pin_memory
        self.train_dataset = EmotionDataset(X_train, y_train, self.tokenizer, max_len=self.max_seq_length)
        self.val_dataset = EmotionDataset(X_val, y_val, self.tokenizer, max_len=self.max_seq_length)
        self.test_dataset = EmotionDataset(X_test, y_test, self.tokenizer, max_len=self.max_seq_length)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size*2, num_workers=0)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size*2, num_workers=0)

    def _init_model(self):
        config = AutoConfig.from_pretrained(self.model_name)
        config.num_labels = len(self.emotional_states)
        config.problem_type = "multi_label_classification"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=config)
        self.model = self.model.to(self.device)
        # Use torch.compile for further speedup if available (PyTorch 2.0+) and not on Windows
        if hasattr(torch, 'compile') and platform.system() != 'Windows':
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass

    def train(self, epochs=2, patience=1, accumulation_steps=4):
        # Use TinyBERT notebook defaults for speed
        self.training_losses = []
        self.validation_losses = []
        self.epoch_times = []
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        total_steps = (len(self.train_loader) // accumulation_steps) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        self._train_model(self.model, self.train_loader, self.val_loader, optimizer, scheduler, self.device, epochs, patience, accumulation_steps)

    def _train_model(self, model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs=2, patience=1, accumulation_steps=4):
        model.train()
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_state = None
        self.training_losses = []
        self.validation_losses = []
        self.epoch_times = []
        for epoch in range(epochs):
            start_time = time_module.time()
            print(f"Epoch {epoch+1}/{epochs}")
            progress_bar = tqdm(train_dataloader, desc="Training", leave=True)
            epoch_loss = 0
            optimizer.zero_grad()
            for i, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['targets'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = torch.nn.BCEWithLogitsLoss()(logits, targets)
                loss = loss / accumulation_steps
                loss.backward()
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                epoch_loss += loss.item() * accumulation_steps
                progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})
            avg_train_loss = epoch_loss / len(train_dataloader)
            self.training_losses.append(avg_train_loss)
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validating"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    targets = batch['targets'].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    batch_loss = torch.nn.BCEWithLogitsLoss()(logits, targets)
                    val_loss += batch_loss.item()
            avg_val_loss = val_loss / len(val_dataloader)
            self.validation_losses.append(avg_val_loss)
            end_time = time_module.time()
            self.epoch_times.append(end_time - start_time)
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
                best_model_state = {k: v.cpu().detach() for k, v in model.state_dict().items()}
            else:
                early_stop_counter += 1
                print(f"Early stopping counter: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        training_time = sum(self.epoch_times)
        print(f"Training completed in {training_time:.2f} seconds")
        if best_model_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    def get_training_stats(self):
        return {
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses,
            'epoch_times': self.epoch_times
        }

    def evaluate(self):
        self.model.eval()
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['targets']
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
                all_predictions.append(predictions)
                all_targets.append(targets.numpy())
        predictions = np.vstack(all_predictions)
        actual_labels = np.vstack(all_targets)
        return predictions, actual_labels

    def report(self, predictions, actual_labels):
        accuracies = {}
        reports = {}
        for i, emotion in enumerate(self.emotional_states):
            acc = accuracy_score(actual_labels[:, i], predictions[:, i])
            accuracies[emotion] = acc
        overall_acc = accuracy_score(actual_labels.flatten(), predictions.flatten())
        for i, emotion in enumerate(self.emotional_states):
            unique_actual = np.unique(actual_labels[:, i])
            unique_pred = np.unique(predictions[:, i])
            if len(unique_actual) == 1 and len(unique_pred) == 1:
                only_class = unique_actual[0]
                report = {
                    'note': f"Only one class present in both actual and predicted: {only_class}",
                    'all_samples': 'Not Present' if only_class == 0 else 'Present',
                    'classification_report': None
                }
            elif len(unique_actual) == 1 or len(unique_pred) == 1:
                label_val = unique_actual[0] if len(unique_actual) == 1 else unique_pred[0]
                report = {
                    'note': f"Only one class present: {'Present' if label_val == 1 else 'Not Present'}",
                    'classification_report': classification_report(
                        actual_labels[:, i], 
                        predictions[:, i], 
                        labels=[0, 1],
                        target_names=['Not Present', 'Present'],
                        zero_division=0,
                        output_dict=True
                    )
                }
            else:
                report = {
                    'classification_report': classification_report(
                        actual_labels[:, i], 
                        predictions[:, i], 
                        target_names=['Not Present', 'Present'],
                        output_dict=True
                    )
                }
            reports[emotion] = report
        metrics = {
            'emotion_accuracies': accuracies,
            'overall_accuracy': overall_acc,
            'classification_reports': reports
        }
        return metrics
    
    @staticmethod
    def get_hyperparameter_grid():
        return HYPERPARAMETERS['tinybert']
    
    @staticmethod
    def train_model_with_stats(model, train_loader, val_loader, optimizer, scheduler, device, epochs=2, patience=1, accumulation_steps=4):
        train_losses = []
        val_losses = []
        epoch_times = []
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_state = None
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        for epoch in range(epochs):
            start_time = time_module.time()
            model.train()
            epoch_loss = 0
            optimizer.zero_grad()
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=True)
            for i, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                targets = batch['targets'].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = torch.nn.BCEWithLogitsLoss()(logits, targets)
                    loss = loss / accumulation_steps
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                epoch_loss += loss.item() * accumulation_steps
                progress_bar.set_postfix({'loss': loss.item() * accumulation_steps})
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}"):
                    input_ids = batch['input_ids'].to(device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                    targets = batch['targets'].to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                        batch_loss = torch.nn.BCEWithLogitsLoss()(logits, targets)
                val_loss += batch_loss.item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            epoch_times.append(time_module.time() - start_time)
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
                best_model_state = {k: v.cpu().detach() for k, v in model.state_dict().items()}
            else:
                early_stop_counter += 1
                print(f"Early stopping counter: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        if best_model_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        return train_losses, val_losses, best_val_loss, epoch_times

    @staticmethod
    def evaluate_model(model, test_loader, device):
        model.eval()
        all_predictions = []
        all_targets = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['targets']
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
                all_predictions.append(predictions)
                all_targets.append(targets.numpy())
        predictions = np.vstack(all_predictions)
        actual_labels = np.vstack(all_targets)
        return predictions, actual_labels

    @classmethod
    def run_hyperparameter_tuning(cls, X_train, y_train, X_val, y_val, X_test, y_test, emotional_states, device, tokenizer, max_seq_length):
        # Use TinyBERT notebook hyperparameter grid and settings
        hyperparameters = [
            {'learning_rate': 5e-5, 'batch_size': 16, 'epochs': 2, 'patience': 1, 'accumulation_steps': 4},
            {'learning_rate': 1e-4, 'batch_size': 16, 'epochs': 2, 'patience': 1, 'accumulation_steps': 4},
        ]
        results = []
        for i, params in enumerate(hyperparameters):
            print(f"\n--- Hyperparameter Configuration {i+1}/{len(hyperparameters)} ---")
            print(f"Learning Rate: {params['learning_rate']}")
            print(f"Batch Size: {params['batch_size']}")
            print(f"Max Epochs: {params['epochs']}")
            print(f"Early Stopping Patience: {params['patience']}")
            # Data loaders
            train_dataset = EmotionDataset(X_train, y_train, tokenizer, max_len=32)
            val_dataset = EmotionDataset(X_val, y_val, tokenizer, max_len=32)
            test_dataset = EmotionDataset(X_test, y_test, tokenizer, max_len=32)
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size']*2, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=params['batch_size']*2, num_workers=0)
            # Model
            config = AutoConfig.from_pretrained(NLP_CONFIG['default_model'])
            config.num_labels = len(emotional_states)
            config.problem_type = "multi_label_classification"
            model = AutoModelForSequenceClassification.from_pretrained(NLP_CONFIG['default_model'], config=config)
            model = model.to(device)
            optimizer = AdamW(model.parameters(), lr=params['learning_rate'])
            total_steps = (len(train_loader) // params['accumulation_steps']) + 1
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=total_steps
            )
            # Training
            train_losses, val_losses, best_val_loss, epoch_times = cls.train_model_with_stats(
                model, train_loader, val_loader, optimizer, scheduler, device,
                epochs=params['epochs'], patience=params['patience'], accumulation_steps=params['accumulation_steps']
            )
            # Evaluation
            predictions, actual_labels = cls.evaluate_model(model, test_loader, device)
            overall_acc = accuracy_score(actual_labels.flatten(), predictions.flatten())
            results.append({
                'params': params,
                'accuracy': overall_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'model': model,
                'epoch_times': epoch_times,
                'predictions': predictions,
                'actual_labels': actual_labels
            })
            print(f"Overall Accuracy: {overall_acc:.4f}")
            print(f"Training Time: {sum(epoch_times):.2f} seconds")
        return results

    @staticmethod
    def get_best_model_from_results(results, weight_accuracy=0.7, weight_time=0.3):
        if not results:
            raise ValueError("Results list is empty.")
        # Normalize times (lower is better)
        times = [result.get('training_time', sum(result.get('epoch_times', []))) for result in results]
        max_time = max(times)
        min_time = min(times)
        time_range = max_time - min_time if max_time > min_time else 1
        normalized_times = [1 - ((t - min_time) / time_range) for t in times]
        # Calculate combined score (higher is better)
        combined_scores = [weight_accuracy * result['accuracy'] + weight_time * norm_time 
                          for result, norm_time in zip(results, normalized_times)]
        best_idx = int(np.argmax(combined_scores))
        best_model = results[best_idx]['model']
        best_params = results[best_idx]['params']
        return best_model, best_params, best_idx, combined_scores

    @staticmethod
    def predict_emotions_raw(text, model, tokenizer, device, threshold=0.5):
        model.eval()
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=32,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            preds = (probs >= threshold)
        # Always use EMOTION_STATES for output keys
        from src.config import EMOTION_STATES
        return {emotion: bool(pred) for emotion, pred in zip(EMOTION_STATES, preds)}
        
    @staticmethod
    def predict_emotions(text, model, tokenizer, device, emotion_variations_path=None, negation_patterns_path=None):
        post_processor = EmotionPostProcessor(
            emotion_variations_path=emotion_variations_path,
            negation_patterns_path=negation_patterns_path
        )
        return post_processor.predict(text, model, tokenizer, device)
    
    
    @staticmethod
    def evaluate_model_with_post_processing(model, test_loader, tokenizer, device, emotion_variations_path=None, negation_patterns_path=None):
        model.eval()
        post_processor = EmotionPostProcessor(
            emotion_variations_path=emotion_variations_path,
            negation_patterns_path=negation_patterns_path
        )
        all_predictions = []
        all_targets = []
        all_texts = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating with Post-Processing"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['targets']
                texts = batch['text']
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                post_processed_predictions = []
                for i, text in enumerate(texts):
                    emotion_results = post_processor.predict(text, model, tokenizer, device)
                    post_processed_row = [1 if emotion_results[emotion] else 0 for emotion in post_processor.emotional_states]
                    post_processed_predictions.append(post_processed_row)
                all_predictions.extend(post_processed_predictions)
                all_targets.extend(targets.numpy())
                all_texts.extend(texts)
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        accuracies = {}
        for i, emotion in enumerate(post_processor.emotional_states):
            acc = accuracy_score(all_targets[:, i], all_predictions[:, i])
            accuracies[emotion] = acc
        overall_acc = accuracy_score(all_targets.flatten(), all_predictions.flatten())
        return {
            'accuracy': overall_acc,
            'emotion_accuracies': accuracies,
            'predictions': all_predictions,
            'targets': all_targets,
            'texts': all_texts
        }
    
    @staticmethod
    def export_best_model(model, tokenizer, export_dir):
        os.makedirs(export_dir, exist_ok=True)
        model.save_pretrained(export_dir)
        tokenizer.save_pretrained(export_dir)
        print(f"Best model and tokenizer exported to: {export_dir}")
    
    @staticmethod
    def load_from_pretrained(model_dir, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # Return a simple object with .model and .tokenizer attributes
        class ModelWithTokenizer:
            pass
        mwt = ModelWithTokenizer()
        mwt.model = model
        mwt.tokenizer = tokenizer
        return mwt