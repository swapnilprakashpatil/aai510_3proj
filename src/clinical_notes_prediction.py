import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
import numpy as np
from src.config import RANDOM_STATE, NLP_CONFIG, HYPERPARAMETERS

class NoShowPredictor:
    def __init__(self, config, device='cpu', seed=RANDOM_STATE):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.bert = AutoModel.from_pretrained(config["model_name"])
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 1)
        self.device = torch.device(device)
        self.bert = self.bert.to(self.device)
        self.classifier = self.classifier.to(self.device)

    def get_embeddings(self, texts):
        batch_size = self.config.get('batch_size', 2)
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = list(texts)[i:i+batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config["max_length"],
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.bert(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu()
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0)

    def train(self, df, text_col="AssessmentNotes", label_col="No-show", epochs=None, lr=None):
        if epochs is None:
            epochs = self.config.get("epochs", 3)
        if lr is None:
            lr = self.config.get("learning_rate", 2e-5)
        print(f"Training NoShowPredictor for {epochs} epochs with learning rate {lr}...")
        X = df[text_col]
        y = df[label_col].astype(np.float32).values
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        train_emb = self.get_embeddings(X_train)
        val_emb = self.get_embeddings(X_val)
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for epoch in range(epochs):
            self.classifier.train()
            optimizer.zero_grad()
            outputs = self.classifier(train_emb).squeeze()
            loss = loss_fn(outputs, torch.tensor(y_train, device=self.device))
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{epochs} - Training loss: {loss.item():.4f}")
        self.classifier.eval()
        with torch.no_grad():
            val_outputs = self.classifier(val_emb).squeeze()
            val_preds = torch.sigmoid(val_outputs).cpu().numpy() > 0.5
        acc = accuracy_score(y_val, val_preds)
        print(f"Validation Accuracy: {acc:.4f}")
        return acc

    def evaluate(self, df, text_col="AssessmentNotes", label_col="No-show"):
        print("Evaluating NoShowPredictor...")
        X = df[text_col]
        y = df[label_col].astype(np.float32).values
        emb = self.get_embeddings(X)
        with torch.no_grad():
            outputs = self.classifier(emb).squeeze()
            preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
        acc = accuracy_score(y, preds)
        cm = confusion_matrix(y, preds)
        roc = roc_auc_score(y, torch.sigmoid(outputs).cpu().numpy())
        print(classification_report(y, preds))
        self.plotter.plot_accuracy(y, preds)
        self.plotter.plot_confusion_matrix(y, preds)
        self.plotter.plot_roc_auc(y, torch.sigmoid(outputs).cpu().numpy())
        print(f"Evaluation complete. Accuracy: {acc:.4f}, ROC AUC: {roc:.4f}")
        return acc, cm, roc

    def predict(self, texts):
        emb = self.get_embeddings(texts)
        with torch.no_grad():
            outputs = self.classifier(emb).squeeze()
            preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
        return preds

    def tune_hyperparameters(self, df, text_col="AssessmentNotes", label_col="No-show"):
        best_acc = 0
        best_lr = None
        best_epochs = None
        print("Starting hyperparameter tuning...")
        for params in HYPERPARAMETERS.get('tinybert', []):
            lr = params['learning_rate']
            epochs = params['epochs']
            print(f"Testing learning_rate={lr}, epochs={epochs}")
            acc = self.train(df, text_col, label_col, epochs=epochs, lr=lr)
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
                best_epochs = epochs
        print(f"Best learning rate: {best_lr}, Best epochs: {best_epochs}, Best accuracy: {best_acc:.4f}")
        return best_lr, best_epochs

    def save(self, path):
        torch.save(self.classifier.state_dict(), path)
