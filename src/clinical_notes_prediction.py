import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.config import NLP_CONFIG, HYPERPARAMETERS, RANDOM_STATE, is_step_enabled
from src.plots import PlotGenerator

class ClinicalNotesNoShowPredictor:
    def __init__(self, config=None):
        if config is None:
            config = HYPERPARAMETERS['tiny_clinicalbert'][0]
        self.config = config
        self.model_name = config.get('model_name', NLP_CONFIG['default_model'])
        self.max_length = config.get('max_length', NLP_CONFIG['max_length'])
        self.epochs = config.get('epochs', NLP_CONFIG['epochs'])
        self.batch_size = config.get('batch_size', NLP_CONFIG['batch_size'])
        self.learning_rate = config.get('learning_rate', NLP_CONFIG['learning_rate'])
        self.weight_decay = config.get('weight_decay', NLP_CONFIG.get('weight_decay', 0.01))
        self.warmup_ratio = config.get('warmup_ratio', NLP_CONFIG.get('warmup_ratio', 0.1))
        self.eval_steps = config.get('eval_steps', NLP_CONFIG.get('eval_steps', 10))
        self.save_steps = config.get('save_steps', NLP_CONFIG.get('save_steps', 10))
        self.logging_steps = config.get('logging_steps', NLP_CONFIG.get('logging_steps', 10))
        self.device = NLP_CONFIG.get('device', 'cpu')
        self.seed = NLP_CONFIG.get('seed', RANDOM_STATE)
        self.num_labels = NLP_CONFIG.get('num_labels', 2)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)

    def prepare_data(self, df, text_col="AssessmentNotes", label_col="No-show"):
        df["labels"] = df[label_col].astype(int)
        dataset = Dataset.from_pandas(df[[text_col, "labels"]])
        splits = dataset.train_test_split(test_size=0.2, seed=self.seed)
        train_ds, valid_ds = splits["train"], splits["test"]
        train_ds = train_ds.map(lambda batch: self.tokenizer(batch[text_col], padding="max_length", truncation=True, max_length=self.max_length), batched=True, load_from_cache_file=False)
        valid_ds = valid_ds.map(lambda batch: self.tokenizer(batch[text_col], padding="max_length", truncation=True, max_length=self.max_length), batched=True, load_from_cache_file=False)
        train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        valid_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        return train_ds, valid_ds

    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    def get_training_args(self, output_dir="clinicalbert_model"):
        return TrainingArguments(
                output_dir=output_dir,
                no_cuda=True,
                use_cpu=True,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size * 2,
                dataloader_num_workers=os.cpu_count(),
                eval_strategy="steps", 
                eval_steps=self.eval_steps,
                save_strategy="steps",
                save_steps=self.save_steps,
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
                save_total_limit=2,
                num_train_epochs=self.epochs,
                learning_rate=self.learning_rate,
                logging_steps=self.logging_steps,
                remove_unused_columns=False,
            )


    def train(self, train_ds, valid_ds, output_dir="clinicalbert_model"):
        training_args = self.get_training_args(output_dir)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        trainer.train()
        self.model = trainer.model
        return trainer

    def evaluate(self, trainer, valid_ds):
        return trainer.evaluate(valid_ds)

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        with torch.no_grad():
            outputs = self.model(**inputs)
            preds = outputs.logits.argmax(-1).cpu().numpy()
        return preds

    def save(self, output_dir="clinicalbert_model"):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def plot_results(self, training_stats=None, y_true=None, y_pred=None):
        plotter = PlotGenerator()
        if training_stats and 'training_loss' in training_stats and 'validation_loss' in training_stats:
            plotter.plot_training_validation_loss(training_stats['training_loss'], training_stats['validation_loss'])
        if y_true is not None and y_pred is not None:
            PlotGenerator.plot_accuracy(y_true, y_pred)
            PlotGenerator.plot_confusion_matrix(y_true, y_pred)