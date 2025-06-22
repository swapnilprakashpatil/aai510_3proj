import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter
import os
import re
from wordcloud import WordCloud
from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
)

from typing import Optional, Tuple
from src.config import EMOTION_STATES

import nltk
from nltk.corpus import stopwords

try:
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. Using basic stopwords.")

class PlotGenerator:
    
    def __init__(self, style: str = 'whitegrid', palette: str = 'viridis', figsize: Tuple[int, int] = (10, 6)):
        self.default_style = style
        self.default_palette = palette
        self.default_figsize = figsize
        sns.set_style(self.default_style)

    def _setup_plot(self, figsize: Optional[Tuple[int, int]] = None, style: Optional[str] = None, palette: Optional[str] = None):
        if figsize is None:
            figsize = self.default_figsize
        if style:
            sns.set_style(style)
        if palette:
            sns.set_palette(palette)

        plt.figure(figsize=figsize)

    def _add_common_plot_elements(self, title: str, xlabel: str, ylabel: str):
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(xlabel, fontsize=13)
        plt.ylabel(ylabel, fontsize=13)
        plt.xticks(fontsize=11, rotation=45, ha='right')
        plt.yticks(fontsize=11)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    def plot_correlation_heatmap(self, data: pd.DataFrame):
        self._setup_plot(figsize=(10, 8))
        correlation_matrix = data.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='crest', 
                    square=True, linewidths=0.5, linecolor='gray', 
                    cbar_kws={'shrink': 0.8})
        self._add_common_plot_elements('Correlation Heatmap', '', '')
        plt.tight_layout()
        plt.show()

    def plot_countplot(self, data: pd.DataFrame, column: str, title: str = '', xlabel: str = '', ylabel: str = '', 
                       figsize: Tuple[int, int] = (8, 5), palette: str = 'deep', colors: Optional[list] = None):
        
        self._setup_plot(figsize=figsize)
        sns.set_style('darkgrid')
        color_param = {'palette': colors if colors else palette}

        ax = sns.countplot(x=column, data=data, edgecolor='black', linewidth=0.8, alpha=0.85, **color_param)
        ax.set_title(title or f'{column} Countplot', fontsize=16, fontweight='bold')
        ax.set_xlabel(xlabel or column, fontsize=13)
        ax.set_ylabel(ylabel or 'Count', fontsize=13)
        ax.tick_params(axis='both', labelsize=11)
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', fontsize=11, color='black', 
                       xytext=(0, 8), textcoords='offset points')
        plt.tight_layout()
        plt.show()

    def plot_boxplot(self, data: pd.DataFrame, x: str, y: str):
        self._setup_plot(figsize=(10, 6))
        sns.boxplot(data=data, x=x, y=y, palette='crest')
        self._add_common_plot_elements(f'Box Plot of {y} by {x}', x, y)
        plt.tight_layout()
        plt.show()

    def plot_violinplot(self, data: pd.DataFrame, x: str, y: str):
        self._setup_plot(figsize=(10, 6))
        sns.violinplot(data=data, x=x, y=y, palette='crest')
        self._add_common_plot_elements(f'Violin Plot of {y} by {x}', x, y)
        plt.tight_layout()
        plt.show()

    def plot_pca_scatter(self, data, pca_results, cluster_labels):
        self._setup_plot(figsize=(10, 8))
        plt.scatter(pca_results[:, 0], pca_results[:, 1], c=cluster_labels, 
                   cmap='viridis', alpha=0.5)
        plt.title('PCA Scatter Plot', fontsize=16, fontweight='bold')
        plt.xlabel('Principal Component 1', fontsize=13)
        plt.ylabel('Principal Component 2', fontsize=13)
        plt.colorbar(label='Cluster Label')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_word_cloud(self, word_freq):
        self._setup_plot(figsize=(10, 6))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_histplot(self, data, column, bins=30, kde=True, title='', xlabel='', 
                     ylabel='', figsize=(10, 6), color=None):
        self._setup_plot(figsize=figsize)
        ax = sns.histplot(data[column], bins=bins, kde=kde, color=color, 
                         edgecolor=None, linewidth=0, alpha=0.85)
        median = data[column].median()
        mean = data[column].mean()
        ax.axvline(median, color='red', linestyle='--', linewidth=2, 
                  label=f'Median: {median:.2f}')
        ax.axvline(mean, color='blue', linestyle='-', linewidth=2, 
                  label=f'Mean: {mean:.2f}')
        ax.legend(fontsize=11)
        ax.set_title(title or f'{column} Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel(xlabel or column, fontsize=13)
        ax.set_ylabel(ylabel or 'Frequency', fontsize=13)
        ax.tick_params(axis='both', labelsize=11)
        plt.tight_layout()
        plt.show()

    def plot_countplot_advanced(self, data, column, title='', xlabel='', ylabel='', 
                               figsize=(8, 5), palette='deep', colors=None):
        
        self._setup_plot(figsize=figsize)
        sns.set_style('darkgrid')
        color_param = {'palette': colors if colors else palette}
        
        ax = sns.countplot(x=column, data=data, edgecolor='black', linewidth=0.8, 
                          alpha=0.85, **color_param)
        ax.set_title(title or f'{column} Countplot', fontsize=16, fontweight='bold')
        ax.set_xlabel(xlabel or column, fontsize=13)
        ax.set_ylabel(ylabel or 'Count', fontsize=13)
        ax.tick_params(axis='both', labelsize=11)
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', fontsize=11, color='black', 
                       xytext=(0, 8), textcoords='offset points')
        plt.tight_layout()
        plt.show()

    def plot_heatmap(self, data, title='', fmt='.2f', cmap='viridis', 
                    square=True, figsize=(12, 8)):
        
        self._setup_plot(figsize=figsize)
        sns.set_palette('deep')
        sns.set_style('darkgrid')
        ax = sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap, square=square, 
                        linewidths=0, linecolor=None, cbar_kws={'shrink': 0.8})
        ax.set_title(title or 'Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.xticks(fontsize=11, rotation=45, ha='right')
        plt.yticks(fontsize=11)
        plt.tight_layout()
        plt.show()

    def plot_emotional_states_bar(self, df, emotional_states=None, figsize=(8, 5), palette='deep'):
        
        if emotional_states is None:
            emotional_states = EMOTION_STATES
            
        emotion_counts = df[emotional_states].sum().sort_values(ascending=False)
        self._setup_plot(figsize=figsize)
        sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette=palette)        
        plt.title('Emotional States in Patient Sentiment', fontsize=16, fontweight='bold')
        plt.xlabel('Emotional State', fontsize=13)
        plt.ylabel('Count', fontsize=13)
        plt.tight_layout()
        plt.show()

    def plot_text_wordcloud(self, series, title='Word Cloud', figsize=(10, 6), remove_stopwords=True):
        # Get stopwords using NLTK or fallback to CSV-based set
        if remove_stopwords:
            if NLTK_AVAILABLE:
                stop_words = set(stopwords.words('english'))
                # Add stopwords from CSV file
                csv_stopwords = self.get_default_stopwords()
                stop_words = stop_words.union(csv_stopwords)
            else:
                # Use CSV-based stopwords as fallback
                stop_words = self.get_default_stopwords()
        else:
            stop_words = set()
        
        # Process text: join all text, convert to lowercase, remove punctuation
        text = ' '.join(series.dropna().astype(str))
        text = text.lower()
        
        # Remove punctuation and split into words
        # Keep only alphabetic characters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        words = text.split()
        
        # Filter out stopwords and very short words (less than 3 characters)
        if remove_stopwords:
            filtered_words = [word for word in words if word not in stop_words and len(word) >= 3]
        else:
            filtered_words = [word for word in words if len(word) >= 3]
        
        # Create word frequency counter
        word_freq = Counter(filtered_words)
        
        # Remove words that appear only once (optional, can be commented out)
        word_freq = Counter({word: freq for word, freq in word_freq.items() if freq > 1})
        
        if not word_freq:
            print("No words remaining after filtering. Try setting remove_stopwords=False or check your data.")
            return
        
        self._setup_plot(figsize=figsize)

        # Create word cloud with better parameters
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            relative_scaling=0.5,
            colormap='viridis',
            stopwords=stop_words if remove_stopwords else None
        ).generate_from_frequencies(word_freq)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print some statistics
        print(f"Total unique words after filtering: {len(word_freq)}")
        print(f"Top 10 most frequent words: {dict(word_freq.most_common(10))}")

    def plot_confusion_matrix(self, conf_matrix, classes=None, title='Confusion Matrix', 
                             figsize=(8, 6), cmap='Blues', normalize=False):
        
        self._setup_plot(figsize=figsize)
        
        if normalize:
            conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        ax = sns.heatmap(conf_matrix, annot=True, fmt=fmt, cmap=cmap)
        
        if classes:
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=13)
        plt.xlabel('Predicted Label', fontsize=13)
        plt.tight_layout()
        plt.show()
        
    def plot_training_metrics(self, metrics, title='Training Metrics', figsize=(10, 6)):
        
        self._setup_plot(figsize=figsize)
        
        for metric_name, values in metrics.items():
            if isinstance(values, list) and len(values) > 0:
                plt.plot(range(1, len(values) + 1), values, marker='o', label=metric_name)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=13)
        plt.ylabel('Score', fontsize=13)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_emotion_distribution(self, data, emotions=None, figsize=(10, 6), 
                                 title='Emotion Distribution', palette='viridis', by_category=None):
        
        if emotions is None:
            emotions = EMOTION_STATES
        
        self._setup_plot(figsize=figsize)
        
        if by_category is not None and by_category in data.columns:
            emotion_counts = {}
            categories = data[by_category].unique()
            
            for category in categories:
                category_data = data[data[by_category] == category]
                emotion_counts[category] = category_data[emotions].sum().values
            
            bar_width = 0.8 / len(categories)
            r = np.arange(len(emotions))
            
            for i, (category, counts) in enumerate(emotion_counts.items()):
                plt.bar(r + i * bar_width, counts, width=bar_width, 
                       label=category, alpha=0.7)
            
            plt.xticks(r + bar_width * (len(categories) - 1) / 2, emotions)
            plt.legend(title=by_category)
            
        else:
            emotion_counts = data[emotions].sum().sort_values(ascending=False)
            sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette=palette)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Emotion', fontsize=13)
        plt.ylabel('Count', fontsize=13)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_emotion_heatmap(self, data, emotions=None, correlation_with=None, 
                            figsize=(10, 8), cmap='coolwarm'):
        
        if emotions is None:
            emotions = EMOTION_STATES
        
        if correlation_with is None:
            corr_matrix = data[emotions].corr()
        else:
            numeric_cols = data[correlation_with].select_dtypes(include=['number']).columns.tolist()
            if not numeric_cols:
                print("No numeric columns found in correlation_with. Correlating emotions with each other.")
                corr_matrix = data[emotions].corr()
            else:
                corr_matrix = data[emotions + numeric_cols].corr()
                corr_matrix = corr_matrix.loc[emotions, numeric_cols]
        
        self._setup_plot(figsize=figsize)
        mask = np.zeros_like(corr_matrix, dtype=bool)
        
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, mask=mask, square=True,
                    linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Emotion Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_sentiment_distribution(self, df, sentiment_col='sentiment', by_column=None, 
                                   figsize=(10, 6), title='Sentiment Distribution'):
        
        self._setup_plot(figsize=figsize)
        
        if by_column and by_column in df.columns:
            cross_tab = pd.crosstab(df[by_column], df[sentiment_col])
            cross_tab.plot(kind='bar', stacked=True, colormap='viridis')
            plt.xlabel(by_column, fontsize=13)
            plt.ylabel('Count', fontsize=13)
        else:
            value_counts = df[sentiment_col].value_counts().sort_index()
            ax = sns.barplot(x=value_counts.index, y=value_counts.values, palette='viridis')
            plt.xlabel('Sentiment', fontsize=13)
            plt.ylabel('Count', fontsize=13)
            
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', 
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', fontsize=11, color='black', 
                           xytext=(0, 8), textcoords='offset points')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_sentiment_emotion_heatmap(self, df, emotions=None, figsize=(10, 8), 
                                      cmap='coolwarm', title='Sentiment-Emotion Correlation'):
        
        if emotions is None:
            emotions = EMOTION_STATES
        
        corr_data = pd.DataFrame()
        corr_data['sentiment_score'] = df['sentiment'].map({'negative': 0, 'positive': 1})
        
        for emotion in emotions:
            if emotion in df.columns:
                corr_data[emotion] = df[emotion]
        
        corr_matrix = corr_data.corr()
        
        self._setup_plot(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, linewidths=0.5,
                    cbar_kws={"shrink": .8})
        
        sentiment_label = 'Sentiment\n(0=negative, 1=positive)'
        corr_matrix.index = [sentiment_label if x == 'sentiment_score' else x for x in corr_matrix.index]
        corr_matrix.columns = [sentiment_label if x == 'sentiment_score' else x for x in corr_matrix.columns]
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_bert_training_progress(self, training_stats, figsize=(12, 5)):
        stats_df = pd.DataFrame(training_stats)
        
        self._setup_plot(figsize=figsize)
        
        plt.subplot(1, 2, 1)
        plt.plot(stats_df['epoch'], stats_df['training_loss'], marker='o', 
                linestyle='-', color='blue')
        plt.title('Training Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if 'val_accuracy' in stats_df.columns:
            plt.subplot(1, 2, 2)
            plt.plot(stats_df['epoch'], stats_df['val_accuracy'], marker='o', 
                    linestyle='-', label='Accuracy', color='green')
            
            if 'val_f1' in stats_df.columns:
                plt.plot(stats_df['epoch'], stats_df['val_f1'], marker='s', 
                        linestyle='-', label='F1 Score', color='orange')
            
            plt.title('Validation Metrics', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()

    def plot_emotion_by_sentiment(self, df, emotions=None, figsize=(12, 8), 
                                 title='Emotions by Sentiment'):
        
        if emotions is None:
            emotions = ['anxiety', 'stress', 'confusion', 'hopeful', 'fear']
        
        self._setup_plot(figsize=figsize)
        
        if not set(emotions).issubset(df.columns):
            for emo in emotions:
                df[emo] = df['emotions_detected'].apply(
                    lambda d: 1 if isinstance(d, dict) and d.get(emo) else 0
                )
        group_col = 'sentiment_label' if 'sentiment_label' in df.columns else 'sentiment'
        grouped = df.groupby(group_col)[emotions].mean()
        
        sns.heatmap(grouped, annot=True, fmt=".2f", cmap='viridis', linewidths=0.5)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Emotion', fontsize=13)
        plt.ylabel('Sentiment', fontsize=13)
        plt.tight_layout()
        plt.show()

    def plot_sentiment_confidence(self, df, figsize=(10, 6), 
                                 title='Sentiment Prediction Confidence'):
        
        if 'confidence' not in df.columns:
            print("No confidence scores available to plot")
            return
        
        self._setup_plot(figsize=figsize)
        
        grouped = df.groupby('sentiment_label')['confidence'].mean().reset_index()
        
        sns.barplot(x='sentiment_label', y='confidence', data=grouped, palette='viridis')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Sentiment', fontsize=13)
        plt.ylabel('Average Confidence Score', fontsize=13)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

    def plot_noshow_rates_by_sentiment(self, noshow_by_sentiment, title='No-Show Rate by Sentiment', 
                                      figsize=(10, 6), palette='viridis'):
        
        self._setup_plot(figsize=figsize)
        sns.barplot(x=noshow_by_sentiment.index, y=noshow_by_sentiment.values, palette=palette)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Sentiment', fontsize=13)
        plt.ylabel('No-Show Rate', fontsize=13)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

    def plot_noshow_rates_by_emotion(self, noshow_by_emotion, title='No-Show Rate by Emotion', 
                                    figsize=(12, 6), palette='viridis'):
        
        self._setup_plot(figsize=figsize)
        sns.barplot(x=noshow_by_emotion.index, y=noshow_by_emotion.values, palette=palette)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Emotion', fontsize=13)
        plt.ylabel('No-Show Rate', fontsize=13)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()

    def plot_sentiment_noshow_analysis(self, analysis_results, figsize=(15, 10)):
        self._setup_plot(figsize=figsize)
        
        # Plot no-show rates by sentiment
        plt.subplot(2, 2, 1)
        noshow_by_sentiment = analysis_results['noshow_by_sentiment']
        sns.barplot(x=noshow_by_sentiment.index, y=noshow_by_sentiment.values, palette='viridis')
        plt.title('No-Show Rate by Sentiment', fontsize=14, fontweight='bold')
        plt.xlabel('Sentiment', fontsize=12)
        plt.ylabel('No-Show Rate', fontsize=12)
        plt.ylim(0, 1)
        
        # Plot no-show rates by emotion
        plt.subplot(2, 2, 2)
        noshow_by_emotion = analysis_results['noshow_by_emotion']
        sns.barplot(x=noshow_by_emotion.index, y=noshow_by_emotion.values, palette='viridis')
        plt.title('No-Show Rate by Dominant Emotion', fontsize=14, fontweight='bold')
        plt.xlabel('Dominant Emotion', fontsize=12)
        plt.ylabel('No-Show Rate', fontsize=12)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Plot sentiment-emotion heatmap
        plt.subplot(2, 2, 3)
        sentiment_emotion_cross = analysis_results['sentiment_emotion_cross']
        sns.heatmap(sentiment_emotion_cross, annot=True, fmt='d', cmap='viridis')
        plt.title('Sentiment-Emotion Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Emotion', fontsize=12)
        plt.ylabel('Sentiment', fontsize=12)
        
        # Plot sentiment-emotion no-show rates
        plt.subplot(2, 2, 4)
        sentiment_emotion_noshow = analysis_results['sentiment_emotion_noshow']
        sns.heatmap(sentiment_emotion_noshow, annot=True, fmt='.2f', cmap='viridis')
        plt.title('No-Show Rate by Sentiment-Emotion', fontsize=14, fontweight='bold')
        plt.xlabel('Emotion', fontsize=12)
        plt.ylabel('Sentiment', fontsize=12)
        
        plt.tight_layout()
        plt.show()

    def plot_reason_categories(self, data, category_column='reason_category', 
                              title='No-Show Reason Categories', figsize=(10, 6), palette='viridis'):
        
        self._setup_plot(figsize=figsize)
        
        category_counts = data[category_column].value_counts().sort_values(ascending=False)
        
        ax = sns.barplot(x=category_counts.index, y=category_counts.values, palette=palette)
        
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', fontsize=11, color='black', 
                       xytext=(0, 8), textcoords='offset points')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Reason Category', fontsize=13)
        plt.ylabel('Count', fontsize=13)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_reason_category_wordclouds(self, data, reason_column='NoShowReason', 
                                       category_column='reason_category', 
                                       title_prefix='Word Cloud for', figsize=(8, 5)):
        
        categories = data[category_column].unique()
        
        for category in categories:
            category_reasons = data[data[category_column] == category][reason_column]
            if len(category_reasons) > 0:
                self.plot_text_wordcloud(
                    series=category_reasons,
                    title=f'{title_prefix} {category.title()} Category',
                    figsize=figsize
                )

    def plot_category_emotion_relationship(self, data, category_column='reason_category', 
                                         emotion_column='dominant_emotion',
                                         title='Relationship Between Reason Categories and Emotions', 
                                         figsize=(12, 8), cmap='viridis'):
        
        if emotion_column not in data.columns:
            print(f"Column '{emotion_column}' not found in the data.")
            return
        
        category_emotion_counts = pd.crosstab(data[category_column], data[emotion_column])
        
        self._setup_plot(figsize=figsize)
        sns.heatmap(category_emotion_counts, annot=True, fmt='d', cmap=cmap)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Emotion', fontsize=13)
        plt.ylabel('Reason Category', fontsize=13)
        plt.tight_layout()
        plt.show()

    def plot_sentiment_analysis_dashboard(self, sentiment_results, figsize=(15, 10)):
        self._setup_plot(figsize=figsize)
        
        # Plot sentiment distribution
        plt.subplot(2, 2, 1)
        self.plot_sentiment_distribution(sentiment_results)
        
        # Plot emotion by sentiment
        plt.subplot(2, 2, 2)
        self.plot_emotion_by_sentiment(sentiment_results)
        
        # Plot sentiment confidence
        plt.subplot(2, 2, 3)
        self.plot_sentiment_confidence(sentiment_results)
        
        # Plot sentiment-emotion heatmap
        plt.subplot(2, 2, 4)
        self.plot_sentiment_emotion_heatmap(sentiment_results)
        
        plt.tight_layout()
        plt.show()

    def plot_tfidf_feature_importance(self, model, feature_names, top_n=20, 
                                     title='Top TF-IDF Features', figsize=(12, 10)):
        
        if not hasattr(model, 'coef_'):
            print("Model does not have 'coef_' attribute. Cannot plot feature importance.")
            return
        
        coefficients = model.coef_[0]
        
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.abs(coefficients)
        })
        
        feature_importance = feature_importance.sort_values('Importance', ascending=False).head(top_n)
        
        self._setup_plot(figsize=figsize)
        sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Coefficient Magnitude', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_tfidf_analysis_dashboard(self, analysis_results, figsize=(15, 12)):
        required_keys = ['model', 'classification_report', 'confusion_matrix', 'feature_names']
        missing_keys = [key for key in required_keys if key not in analysis_results]
        
        if missing_keys:
            print(f"Missing required keys in analysis_results: {', '.join(missing_keys)}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        
        # 1. Plot confusion matrix
        conf_matrix = analysis_results['confusion_matrix']
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Predicted', fontsize=12)
        axes[0, 0].set_ylabel('Actual', fontsize=12)
        
        # 2. Plot classification metrics
        metrics = analysis_results['classification_report']
        class_metrics = {
            'Class': [], 'Precision': [], 'Recall': [], 'F1-Score': [], 'Support': []
        }
        
        for class_label, values in metrics.items():
            if class_label in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            class_metrics['Class'].append(str(class_label))
            class_metrics['Precision'].append(values['precision'])
            class_metrics['Recall'].append(values['recall'])
            class_metrics['F1-Score'].append(values['f1-score'])
            class_metrics['Support'].append(values['support'])
        
        metrics_df = pd.DataFrame(class_metrics)
        metrics_df = metrics_df.melt(
            id_vars=['Class', 'Support'], 
            value_vars=['Precision', 'Recall', 'F1-Score'],
            var_name='Metric', value_name='Value'
        )
        
        sns.barplot(x='Class', y='Value', hue='Metric', data=metrics_df, ax=axes[0, 1])
        axes[0, 1].set_title('Classification Metrics by Class', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylim(0, 1.0)
        axes[0, 1].set_xlabel('Class', fontsize=12)
        axes[0, 1].set_ylabel('Score', fontsize=12)
        
        # 3. Plot feature importance
        feature_importance = pd.DataFrame({
            'Feature': analysis_results['feature_names'],
            'Importance': np.abs(analysis_results['model'].coef_[0])
        })
        
        feature_importance = feature_importance.sort_values('Importance', ascending=False).head(10)
        
        sns.barplot(x='Importance', y='Feature', data=feature_importance,
                   palette='viridis', ax=axes[1, 0])
        axes[1, 0].set_title('Top 10 Features by Importance', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Coefficient Magnitude', fontsize=12)
        axes[1, 0].set_ylabel('Feature', fontsize=12)
        
        # 4. Plot overall accuracy
        accuracy = metrics['accuracy'] if 'accuracy' in metrics else analysis_results.get('accuracy', 0)
        
        axes[1, 1].bar(['Accuracy'], [accuracy], color='teal', alpha=0.7)
        axes[1, 1].set_title('Overall Accuracy', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylim(0, 1.0)
        axes[1, 1].set_ylabel('Score', fontsize=12)
        axes[1, 1].text(0, accuracy / 2, f'{accuracy:.4f}', ha='center', va='center', 
                       fontsize=14, fontweight='bold', color='white')
        
        fig.suptitle('TF-IDF and Logistic Regression Analysis Dashboard', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

    def plot_reason_analysis_dashboard(self, analysis_results, category_map=None, figsize=(15, 12)):
        test_metrics = analysis_results.get('test_metrics', {})
        training_stats = analysis_results.get('training_stats', [])
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        
        # 1. Plot confusion matrix if available
        if 'confusion_matrix' in test_metrics:
            conf_matrix = test_metrics['confusion_matrix']
            
            if category_map:
                labels = [category_map.get(i, f"Category {i}") for i in range(len(category_map))]
            else:
                labels = [f"Category {i}" for i in range(conf_matrix.shape[0])]
            
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels, ax=axes[0, 0])
            axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Predicted', fontsize=12)
            axes[0, 0].set_ylabel('Actual', fontsize=12)
        else:
            axes[0, 0].text(0.5, 0.5, "Confusion Matrix Not Available", 
                           ha='center', va='center', fontsize=14)
            axes[0, 0].axis('off')
        
        # 2. Plot training metrics if available
        if training_stats and isinstance(training_stats, list):
            train_loss = [stat.get('loss', 0) for stat in training_stats]
            val_acc = [stat.get('accuracy', 0) for stat in training_stats]
            epochs = range(1, len(train_loss) + 1)
            
            ax2 = axes[0, 1]
            ax2.plot(epochs, train_loss, 'b-', label='Training Loss')
            ax2.set_title('Training Metrics', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12, color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            
            ax2b = ax2.twinx()
            ax2b.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
            ax2b.set_ylabel('Accuracy', fontsize=12, color='r')
            ax2b.tick_params(axis='y', labelcolor='r')
            
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2b.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
        else:
            axes[0, 1].text(0.5, 0.5, "Training Metrics Not Available", 
                           ha='center', va='center', fontsize=14)
            axes[0, 1].axis('off')
        
        # 3. Plot classification metrics if available
        if 'classification_report' in test_metrics:
            metrics = test_metrics['classification_report']
            class_metrics = {
                'Class': [], 'Precision': [], 'Recall': [], 'F1-Score': [], 'Support': []
            }
            
            for class_label, values in metrics.items():
                if class_label in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                
                if category_map and int(class_label) in category_map:
                    class_name = category_map[int(class_label)]
                else:
                    class_name = f"Category {class_label}"
                    
                class_metrics['Class'].append(class_name)
                class_metrics['Precision'].append(values['precision'])
                class_metrics['Recall'].append(values['recall'])
                class_metrics['F1-Score'].append(values['f1-score'])
                class_metrics['Support'].append(values['support'])
            
            metrics_df = pd.DataFrame(class_metrics)
            
            if not metrics_df.empty:
                metrics_df = metrics_df.melt(
                    id_vars=['Class', 'Support'], 
                    value_vars=['Precision', 'Recall', 'F1-Score'],
                    var_name='Metric', value_name='Value'
                )
                
                sns.barplot(x='Class', y='Value', hue='Metric', data=metrics_df, ax=axes[1, 0])
                axes[1, 0].set_title('Classification Metrics by Class', fontsize=14, fontweight='bold')
                axes[1, 0].set_ylim(0, 1.0)
                axes[1, 0].set_xlabel('Class', fontsize=12)
                axes[1, 0].set_ylabel('Score', fontsize=12)
                if len(class_metrics['Class']) > 3:
                    plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
            else:
                axes[1, 0].text(0.5, 0.5, "Classification Metrics Not Available", 
                              ha='center', va='center', fontsize=14)
                axes[1, 0].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, "Classification Metrics Not Available", 
                           ha='center', va='center', fontsize=14)
            axes[1, 0].axis('off')
        
        # 4. Plot overall accuracy
        accuracy = test_metrics.get('accuracy', 0)
        f1_score = test_metrics.get('f1', 0)
        
        metrics_names = ['Accuracy', 'F1 Score']
        metrics_values = [accuracy, f1_score]
        
        axes[1, 1].bar(metrics_names, metrics_values, color=['teal', 'coral'], alpha=0.7)
        axes[1, 1].set_title('Overall Metrics', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylim(0, 1.0)
        axes[1, 1].set_ylabel('Score', fontsize=12)
        
        for i, v in enumerate(metrics_values):
            axes[1, 1].text(i, v / 2, f'{v:.4f}', ha='center', va='center', 
                           fontsize=14, fontweight='bold', color='white')
        
        fig.suptitle('No-Show Reason Analysis Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

    def print_sentiment_metrics(self, metrics):
        print('Model Accuracy by Emotion:')
        for emotion, acc in metrics['emotion_accuracies'].items():
            print(f"  {emotion}: {acc:.4f}")
        print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.4f}")
        print('\nClassification Reports:')
        for emotion, report in metrics['classification_reports'].items():
            print(f"\n{emotion.capitalize()}:")
            if report.get('note'):
                print(report['note'])
            if report.get('classification_report'):
                for label, scores in report['classification_report'].items():
                    if isinstance(scores, dict):
                        print(f"  {label}: {scores}")

    def plot_accuracy_by_emotion(self, metrics, figsize=(10, 6)):
        emotions = list(metrics['emotion_accuracies'].keys())
        accuracies = list(metrics['emotion_accuracies'].values())
        overall_acc = metrics['overall_accuracy']
        plt.figure(figsize=figsize)
        sns.barplot(x=emotions, y=accuracies, palette='viridis')
        plt.axhline(overall_acc, color='red', linestyle='--', label=f'Overall Accuracy: {overall_acc:.2f}')
        plt.title('Model Accuracy by Emotion')
        plt.xlabel('Emotion')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices(self, actual_labels, predictions, emotions, figsize=(16, 3)):
        n = len(emotions)
        cols = min(n, 5)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(figsize[0], rows * figsize[1]))
        axes = np.array(axes).reshape(-1)
        for i, emotion in enumerate(emotions):
            cm = confusion_matrix(actual_labels[:, i], predictions[:, i])
            ax = axes[i]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_title(f'{emotion.capitalize()}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        plt.suptitle('Confusion Matrices by Emotion', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def plot_training_validation_loss(self, training_losses, validation_losses, figsize=(8, 5)):
        plt.figure(figsize=figsize)
        plt.plot(training_losses, label='Training Loss', marker='o')
        plt.plot(validation_losses, label='Validation Loss', marker='o')
        plt.title('Training and Validation Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_epoch_times(self, epoch_times, figsize=(8, 5)):
        plt.figure(figsize=figsize)
        plt.plot(epoch_times, marker='o')
        plt.title('Time Taken per Epoch (seconds)')
        plt.xlabel('Epoch')
        plt.ylabel('Time (s)')
        plt.tight_layout()
        plt.show()

    def plot_roc_auc_by_emotion(self, actual_labels, predictions, emotion_states, figsize=(10, 8)):
        plt.figure(figsize=figsize)
        for idx, emotion in enumerate(emotion_states):
            fpr, tpr, _ = roc_curve(actual_labels[:, idx], predictions[:, idx])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{emotion} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Each Emotion')
        plt.legend(loc='lower right')
        plt.show()

    @staticmethod
    def plot_accuracy(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        plt.figure(figsize=(5, 3))
        plt.bar(['Accuracy'], [acc], color='skyblue')
        plt.ylim(0, 1)
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    @staticmethod
    def plot_roc_auc(y_true, y_score):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.show()

    def plot_training_metrics(trainer):
        logs = trainer.state.log_history
        train_loss = [x['loss'] for x in logs if 'loss' in x]
        eval_loss = [x['eval_loss'] for x in logs if 'eval_loss' in x]
        plt.figure(figsize=(8,5))
        plt.plot(train_loss, label='Train Loss')
        plt.plot(eval_loss, label='Eval Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

    def plot_wordclouds(self, lda_model, vectorizer, condition, figsize=(None, None), wc_width=None, wc_height=None):
        for idx, topic in enumerate(lda_model.components_):
            fig_w, fig_h = figsize if figsize != (None, None) else (None, None)
            wc_w = wc_width if wc_width else 800
            wc_h = wc_height if wc_height else 400
            if fig_w and fig_h:
                self._setup_plot(figsize=(fig_w, fig_h))
            else:
                plt.figure()
            word_freq = {vectorizer.get_feature_names_out()[i]: topic[i] for i in topic.argsort()[:-21:-1]}
            wordcloud = WordCloud(width=wc_w, height=wc_h, background_color='white').generate_from_frequencies(word_freq)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'{condition} - Topic {idx+1}')
            plt.tight_layout()
            plt.show()

    def plot_bar(self, x, y, title='', ylabel='', xlabel='', figsize=(8, 5), color='skyblue'):
        self._setup_plot(figsize=figsize)
        plt.bar(x, y, color=color)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel(xlabel or 'Category', fontsize=13)
        plt.ylabel(ylabel or 'Value', fontsize=13)
        plt.xticks(fontsize=11, rotation=45, ha='right')
        plt.yticks(fontsize=11)
        plt.tight_layout()
        plt.show()

    def get_default_stopwords(self):
        try:
            # Try to load from CSV file
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            stopwords_file = os.path.join(current_dir, 'data', 'default_stopwords.csv')
            if os.path.exists(stopwords_file):
                df = pd.read_csv(stopwords_file)
                return set(df['word'].str.lower())
        except Exception as e:
            print(f"Warning: Could not load stopwords from CSV file: {e}")

    def plot_text_wordcloud_custom_stopwords(self, series, title='Word Cloud', figsize=(10, 6),
                                            custom_stopwords=None, remove_default_stopwords=True):
        # Start with NLTK stopwords if available, otherwise use CSV-based fallback
        if remove_default_stopwords:
            if NLTK_AVAILABLE:
                default_stopwords = set(stopwords.words('english'))
                # Add stopwords from CSV file
                csv_stopwords = self.get_default_stopwords()
                default_stopwords = default_stopwords.union(csv_stopwords)
            else:
                # Use CSV-based stopwords as fallback
                default_stopwords = self.get_default_stopwords()
        else:
            default_stopwords = set()
        
        # Add custom stopwords if provided
        if custom_stopwords:
            if isinstance(custom_stopwords, (list, tuple)):
                custom_stopwords = set(custom_stopwords)
            default_stopwords.update(custom_stopwords)
        
        # Process text: join all text, convert to lowercase, remove punctuation
        text = ' '.join(series.dropna().astype(str))
        text = text.lower()
        
        # Remove punctuation and split into words
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        words = text.split()
        
        # Filter out stopwords and very short words
        filtered_words = [word for word in words if word not in default_stopwords and len(word) >= 3]
        # Remove words that appear only once
        word_freq = Counter(filtered_words)
        for word in list(word_freq):
            if word_freq[word] <= 1:
                del word_freq[word]
        if not word_freq:
            print("No words remaining after filtering. Try reducing stopwords or check your data.")
            return
        self._setup_plot(figsize=figsize)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            relative_scaling=0.5,
            colormap='viridis',
            stopwords=default_stopwords
        ).generate_from_frequencies(word_freq)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print some statistics
        print(f"Total unique words after filtering: {len(word_freq)}")
        print(f"Top 10 most frequent words: {dict(word_freq.most_common(10))}")

    def plot_pca_3d_colored_by_feature(self, pca_result, df, feature_name, figsize=(8, 12)):
        color_vals = df[feature_name]
        norm_colors = (color_vals.to_numpy() - color_vals.min()) / (color_vals.max() - color_vals.min())
        colors = plt.cm.viridis(norm_colors)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=colors, alpha=0.7)
        ax.set_title(f'PCA Projection (3D) - Colored by {feature_name}', fontsize=14)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax)
        cbar.set_label(feature_name)
        plt.tight_layout()
        plt.show()

    def plot_pca_3d_colored_by_features(self, pca_result, df, feature_names, figsize=(18, 12)):
        n = len(feature_names)
        fig = plt.figure(figsize=(figsize[0]*n, figsize[1]))
        for i, feature_name in enumerate(feature_names, 1):
            color_vals = df[feature_name]
            norm_colors = (color_vals.to_numpy() - color_vals.min()) / (color_vals.max() - color_vals.min())
            colors = plt.cm.viridis(norm_colors)
            ax = fig.add_subplot(1, n, i, projection='3d')
            scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=colors, alpha=0.7)
            ax.set_title(f'{feature_name}', fontsize=20)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, shrink=0.7)
            cbar.set_label(feature_name)
        plt.suptitle('PCA Projection (3D) - Colored by Features', fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def plot_pca_explained_variance(self, explained_df):
        plt.figure(figsize=(10, 6))
        plt.plot(explained_df['Principal Component'], explained_df['Explained Variance Ratio'], marker='o', label='Explained Variance')
        plt.plot(explained_df['Principal Component'], explained_df['Cumulative Variance'], marker='o', linestyle='--', label='Cumulative Variance')
        plt.title('PCA Explained Variance')
        plt.xlabel('Principal Components')
        plt.ylabel('Variance Ratio')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_pca_biplot(self, X_pca, loadings, features, scale=5):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=0.1)
        for i, feature in enumerate(features):
            ax.quiver(0, 0, 0, loadings.iloc[i, 0]*scale, loadings.iloc[i, 1]*scale, loadings.iloc[i, 2]*scale, color='r', arrow_length_ratio=0.1)
            ax.text(loadings.iloc[i, 0]*scale*1.1, loadings.iloc[i, 1]*scale*1.1, loadings.iloc[i, 2]*scale*1.1, feature, color='red')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('3D PCA Biplot (Feature Contributions)')
        plt.tight_layout()
        plt.show()

    def plot_elbow_curve(self, k_values, wcss, optimal_k=None):
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, wcss, marker='o')
        if optimal_k is not None:
            plt.axvline(optimal_k, color='red', linestyle='--', label=f'Elbow at k={optimal_k}')
        plt.title('Elbow Method to Determine Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('WCSS (Inertia)')
        plt.xticks(k_values)
        plt.grid(True)
        if optimal_k is not None:
            plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_clustering_3d(self, X_reduced, labels, k, method='KMeans', cmap='viridis'):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(X_reduced.iloc[:, 0], X_reduced.iloc[:, 1], X_reduced.iloc[:, 2], c=labels, cmap=cmap, alpha=0.7)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'{method} Clusters (k={k})')
        legend = ax.legend(*scatter.legend_elements(), title="Cluster")
        ax.add_artist(legend)
        plt.tight_layout()
        plt.show()

    def plot_clustering_3d_side_by_side(self, X_reduced, kmeans_labels, gmm_labels, k, cmap_kmeans='viridis', cmap_gmm='plasma', figsize=(16, 6)):
        fig = plt.figure(figsize=figsize)
        # KMeans plot
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        scatter1 = ax1.scatter(
            X_reduced[:, 0] if not hasattr(X_reduced, 'iloc') else X_reduced.iloc[:, 0],
            X_reduced[:, 1] if not hasattr(X_reduced, 'iloc') else X_reduced.iloc[:, 1],
            X_reduced[:, 2] if not hasattr(X_reduced, 'iloc') else X_reduced.iloc[:, 2],
            c=kmeans_labels, cmap=cmap_kmeans, alpha=0.7
        )
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_zlabel('PC3')
        ax1.set_title(f'KMeans Clusters (k={k})')
        legend1 = ax1.legend(*scatter1.legend_elements(), title="Cluster")
        ax1.add_artist(legend1)
        # GMM plot
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        scatter2 = ax2.scatter(
            X_reduced[:, 0] if not hasattr(X_reduced, 'iloc') else X_reduced.iloc[:, 0],
            X_reduced[:, 1] if not hasattr(X_reduced, 'iloc') else X_reduced.iloc[:, 1],
            X_reduced[:, 2] if not hasattr(X_reduced, 'iloc') else X_reduced.iloc[:, 2],
            c=gmm_labels, cmap=cmap_gmm, alpha=0.7
        )
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_zlabel('PC3')
        ax2.set_title(f'GMM Clusters (k={k})')
        legend2 = ax2.legend(*scatter2.legend_elements(), title="Cluster")
        ax2.add_artist(legend2)
        plt.tight_layout()
        plt.show()

    def plot_clustering_scores(self, kmeans_score_df, gmm_score_df):
        plt.figure(figsize=(16, 5))
        plt.subplot(1, 3, 1)
        plt.plot(kmeans_score_df['k'], kmeans_score_df['Silhouette Score'], marker='o', color='green', label='KMeans')
        plt.plot(gmm_score_df['k'], gmm_score_df['Silhouette Score'], marker='o', color='blue', label='GMM')
        plt.title('Silhouette Score')
        plt.xlabel('k')
        plt.ylabel('Score')
        plt.grid(True)
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.plot(kmeans_score_df['k'], kmeans_score_df['Davies-Bouldin Score'], marker='o', color='green', label='KMeans')
        plt.plot(gmm_score_df['k'], gmm_score_df['Davies-Bouldin Score'], marker='o', color='blue', label='GMM')
        plt.title('Davies-Bouldin Index')
        plt.xlabel('k')
        plt.ylabel('Score')
        plt.grid(True)
        plt.legend()
        plt.subplot(1, 3, 3)
        plt.plot(kmeans_score_df['k'], kmeans_score_df['Calinski-Harabasz Score'], marker='o', color='green', label='KMeans')
        plt.plot(gmm_score_df['k'], gmm_score_df['Calinski-Harabasz Score'], marker='o', color='blue', label='GMM')
        plt.title('Calinski-Harabasz Score')
        plt.xlabel('k')
        plt.ylabel('Score')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_prediction_model_performance(self, summary_df, show_tuning_impact=False, X_test=None, y_test=None, models_dict=None):
        from sklearn.metrics import precision_recall_curve, average_precision_score
        models = summary_df.index
        f1_scores = summary_df['F1']
        precision_scores = summary_df['Precision']
        recall_scores = summary_df['Recall']
        roc_scores = summary_df['ROC_AUC']
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        if show_tuning_impact:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        # F1 Score comparison
        bars1 = ax1.barh(models, f1_scores, color=colors)
        ax1.set_xlabel('F1 Score')
        ax1.set_title('Model F1 Score Comparison')
        ax1.set_xlim(0, 1)
        for i, v in enumerate(f1_scores):
            ax1.text(v + 0.01, i, f'{v:.3f}', va='center')
        # Precision vs Recall
        scatter = ax2.scatter(recall_scores, precision_scores, c=range(len(models)), 
                             cmap='viridis', s=100, alpha=0.7)
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision vs Recall Trade-off')
        ax2.grid(True, alpha=0.3)
        for i, model in enumerate(models):
            ax2.annotate(str(model).split('(')[0], (recall_scores[i], precision_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        # ROC AUC comparison
        bars3 = ax3.barh(models, roc_scores, color=colors)
        ax3.set_xlabel('ROC AUC')
        ax3.set_title('Model ROC AUC Comparison')
        ax3.set_xlim(0, 1)
        for i, v in enumerate(roc_scores):
            ax3.text(v + 0.01, i, f'{v:.3f}', va='center')

        # Confusion matrix for best model
        confusion_plotted = False
        if X_test is not None and y_test is not None and models_dict is not None:
            best_model_name = summary_df.index[0]
            best_model = models_dict.get(best_model_name)
            if best_model is not None:
                try:
                    y_pred = best_model.predict(X_test)
                    cm = confusion_matrix(y_test, y_pred)
                    if show_tuning_impact:
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax4)
                        ax4.set_title(f'Confusion Matrix: {best_model_name}')
                        ax4.set_xlabel('Predicted')
                        ax4.set_ylabel('Actual')
                        confusion_plotted = True
                    else:
                        plt.figure(figsize=(5, 4))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                        plt.title(f'Confusion Matrix: {best_model_name}')
                        plt.xlabel('Predicted')
                        plt.ylabel('Actual')
                        plt.tight_layout()
                        plt.show()
                        confusion_plotted = True
                except Exception as e:
                    if show_tuning_impact:
                        ax4.text(0.5, 0.5, f"Could not plot confusion matrix for {best_model_name}: {e}",
                                 ha='center', va='center', fontsize=12)
                        ax4.axis('off')
                    else:
                        print(f"Could not plot confusion matrix for {best_model_name}: {e}")
            else:
                if show_tuning_impact:
                    ax4.text(0.5, 0.5, f"Best model '{best_model_name}' not found in models_dict. Skipping confusion matrix plot.",
                             ha='center', va='center', fontsize=12)
                    ax4.axis('off')
                else:
                    print(f"Best model '{best_model_name}' not found in models_dict. Skipping confusion matrix plot.")
        else:
            if show_tuning_impact:
                ax4.text(0.5, 0.5, "", ha='center', va='center', fontsize=12)
                ax4.axis('off')

        # Precision-Recall curves for all models
        if X_test is not None and y_test is not None and models_dict is not None:
            plt.figure(figsize=(8, 6))
            for i, model_name in enumerate(models):
                model = models_dict.get(model_name)
                if model is not None:
                    try:
                        # Try to get probability estimates
                        if hasattr(model, 'predict_proba'):
                            y_scores = model.predict_proba(X_test)[:, 1]
                        elif hasattr(model, 'decision_function'):
                            y_scores = model.decision_function(X_test)
                        else:
                            continue
                        precision, recall, _ = precision_recall_curve(y_test, y_scores)
                        ap = average_precision_score(y_test, y_scores)
                        plt.plot(recall, precision, label=f"{model_name} (AP={ap:.2f})", color=colors[i])
                    except Exception as e:
                        print(f"Could not plot PR curve for {model_name}: {e}")
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves (Test Set)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        if show_tuning_impact:
            # Improvement analysis
            baseline_models = ['Logistic Regression', 'Random Forest', 'XGBoost']
            tuned_models_list = ['Logistic Regression (Tuned)', 'Random Forest (Tuned)', 'XGBoost (Tuned)']
            improvements = []
            model_types = []
            for baseline, tuned in zip(baseline_models, tuned_models_list):
                if baseline in summary_df.index and tuned in summary_df.index:
                    improvement = summary_df.loc[tuned, 'F1'] - summary_df.loc[baseline, 'F1']
                    improvements.append(improvement)
                    model_types.append(baseline.replace(' ', '\n'))
            if len(improvements) > 0:
                fig3, ax5 = plt.subplots(1, 1, figsize=(6, 5))
                bars4 = ax5.bar(model_types, improvements, color=['green' if x > 0 else 'red' for x in improvements])
                ax5.set_ylabel('F1 Score Improvement')
                ax5.set_title('Hyperparameter Tuning Impact')
                ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                for i, v in enumerate(improvements):
                    ax5.text(i, v + 0.001, f'{v:+.3f}', ha='center', va='bottom' if v > 0 else 'top')
        plt.tight_layout()
        plt.show()

    def plot_prediction_feature_importances(self, model, feature_names, title="Feature Importances", top_n=10, color='#4F8FC9'):
        importances = None
        # Try to get feature importances from model
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            # For logistic regression, use absolute value of coefficients
            importances = np.abs(model.coef_)[0]
        else:
            print("Model does not have feature importances or coefficients.")
            return

        # Try to get feature names from model if available
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        # Align feature_names and importances if lengths mismatch
        if len(importances) != len(feature_names):
            min_len = min(len(importances), len(feature_names))
            importances = importances[:min_len]
            feature_names = feature_names[:min_len]

        top_n = min(top_n, len(importances))
        indices = np.argsort(importances)[-top_n:][::-1]
        plt.figure(figsize=(8, 5))
        plt.barh(
            [feature_names[i] for i in indices],
            importances[indices],
            color=color
        )
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(title)
        plt.gca().invert_yaxis()  # Highest importance at the top
        plt.tight_layout()
        plt.show()