import seaborn as sns
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
import pandas as pd

def plot_correlation_heatmap(data):
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='crest', square=True, linewidths=0.5, linecolor='gray', cbar_kws={'shrink': 0.8})
    plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=11, rotation=45, ha='right')
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.show()

def plot_countplot(data, x, hue=None):
    plt.figure(figsize=(10, 6))
    sns.set_palette('crest')
    sns.set_style('whitegrid')
    ax = sns.countplot(data=data, x=x, hue=hue, edgecolor='black', linewidth=1.2, alpha=0.85)
    ax.set_title(f'Count Plot of {x}', fontsize=16, fontweight='bold')
    ax.set_xlabel(x, fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.tick_params(axis='both', labelsize=11)
    plt.tight_layout()
    plt.show()

def plot_boxplot(data, x, y):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x=x, y=y, palette='crest')
    plt.title(f'Box Plot of {y} by {x}', fontsize=16, fontweight='bold')
    plt.xlabel(x, fontsize=13)
    plt.ylabel(y, fontsize=13)
    plt.xticks(fontsize=11, rotation=45, ha='right')
    plt.yticks(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_violinplot(data, x, y):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data, x=x, y=y, palette='crest')
    plt.title(f'Violin Plot of {y} by {x}', fontsize=16, fontweight='bold')
    plt.xlabel(x, fontsize=13)
    plt.ylabel(y, fontsize=13)
    plt.xticks(fontsize=11, rotation=45, ha='right')
    plt.yticks(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_pca_scatter(data, pca_results, cluster_labels):
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_results[:, 0], pca_results[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
    plt.title('PCA Scatter Plot', fontsize=16, fontweight='bold')
    plt.xlabel('Principal Component 1', fontsize=13)
    plt.ylabel('Principal Component 2', fontsize=13)
    plt.colorbar(label='Cluster Label')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_word_cloud(word_freq):
    plt.figure(figsize=(10, 6))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_histplot(data, column, bins=30, kde=True, title='', xlabel='', ylabel='', figsize=(10, 6), color=None):
    plt.figure(figsize=figsize)
    ax = sns.histplot(
        data[column], bins=bins, kde=kde, color=color, edgecolor=None, linewidth=0, alpha=0.85
    )
    median = data[column].median()
    mean = data[column].mean()
    ax.axvline(median, color='red', linestyle='--', linewidth=2, label=f'Median: {median:.2f}')
    ax.axvline(mean, color='blue', linestyle='-', linewidth=2, label=f'Mean: {mean:.2f}')
    ax.legend(fontsize=11)
    ax.set_title(title or f'{column} Distributionssss', fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel or column, fontsize=13)
    ax.set_ylabel(ylabel or 'Frequency', fontsize=13)
    ax.tick_params(axis='both', labelsize=11)
    plt.tight_layout()
    plt.show()

def plot_countplot(data, column, title='', xlabel='', ylabel='', figsize=(8, 5), palette='deep', colors=None):
    plt.figure(figsize=figsize)
    sns.set_style('darkgrid')
    color_param = {'palette': colors if colors else palette}
    
    ax = sns.countplot(x=column, data=data, edgecolor='black', linewidth=0.8, alpha=0.85, **color_param)
    ax.set_title(title or f'{column} Countplot', fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel or column, fontsize=13)
    ax.set_ylabel(ylabel or 'Count', fontsize=13)
    ax.tick_params(axis='both', labelsize=11)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 8), textcoords='offset points')
    plt.tight_layout()
    plt.show()

def plot_heatmap(data, title='', fmt='.2f', cmap='viridis', square=True, figsize=(12, 8)):
    plt.figure(figsize=figsize)
    sns.set_palette('deep')
    sns.set_style('darkgrid')
    ax = sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap, square=square, linewidths=0, linecolor=None, cbar_kws={'shrink': 0.8})
    ax.set_title(title or 'Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=11, rotation=45, ha='right')
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.show()

def plot_emotional_states_bar(df, emotional_states=None, figsize=(8, 5), palette='deep'):
    if emotional_states is None:
        emotional_states = ['anxiety', 'stress', 'confusion', 'hopeful', 'fear']
    emotion_counts = df[emotional_states].sum().sort_values(ascending=False)
    plt.figure(figsize=figsize)
    sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette=palette)
    plt.title('Emotional States in Patient Sentiment', fontsize=16, fontweight='bold')
    plt.xlabel('Emotional State', fontsize=13)
    plt.ylabel('Count', fontsize=13)
    plt.tight_layout()
    plt.show()

def plot_text_wordcloud(series, title='Word Cloud', figsize=(10, 6)):
    words = ' '.join(series.dropna().astype(str)).lower().split()
    word_freq = Counter(words)
    plt.figure(figsize=figsize)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(conf_matrix, classes=None, title='Confusion Matrix', 
                      figsize=(8, 6), cmap='Blues', normalize=False):
    
    plt.figure(figsize=figsize)
    
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    ax = sns.heatmap(conf_matrix, annot=True, fmt=fmt, cmap=cmap)
    
    # Set labels and title
    if classes:
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=13)
    plt.xlabel('Predicted Label', fontsize=13)
    plt.tight_layout()
    plt.show()
    
def plot_training_metrics(metrics, title='Training Metrics', figsize=(10, 6)):
    """
    Plot training metrics over epochs.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing metrics to plot
    title : str, optional
        The title of the plot
    figsize : tuple, optional
        Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    
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

def plot_emotion_distribution(data, emotions=None, figsize=(10, 6), title='Emotion Distribution', 
                             palette='viridis', by_category=None):
    if emotions is None:
        emotions = ['anxiety', 'stress', 'confusion', 'hopeful', 'fear']
    
    plt.figure(figsize=figsize)
    
    if by_category is not None and by_category in data.columns:
        # Create grouped barplots
        emotion_counts = {}
        categories = data[by_category].unique()
        
        for category in categories:
            category_data = data[data[by_category] == category]
            emotion_counts[category] = category_data[emotions].sum().values
        
        # Create bar positions
        bar_width = 0.8 / len(categories)
        r = np.arange(len(emotions))
        
        for i, (category, counts) in enumerate(emotion_counts.items()):
            plt.bar(r + i * bar_width, counts, width=bar_width, label=category, alpha=0.7)
        
        plt.xticks(r + bar_width * (len(categories) - 1) / 2, emotions)
        plt.legend(title=by_category)
        
    else:
        # Simple counts
        emotion_counts = data[emotions].sum().sort_values(ascending=False)
        sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette=palette)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Emotion', fontsize=13)
    plt.ylabel('Count', fontsize=13)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_emotion_heatmap(data, emotions=None, correlation_with=None, figsize=(10, 8), cmap='coolwarm'):
    if emotions is None:
        emotions = ['anxiety', 'stress', 'confusion', 'hopeful', 'fear']
    
    if correlation_with is None:
        # Correlate emotions with each other
        corr_matrix = data[emotions].corr()
    else:
        # Select numeric columns from correlation_with
        numeric_cols = data[correlation_with].select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            print("No numeric columns found in correlation_with. Correlating emotions with each other.")
            corr_matrix = data[emotions].corr()
        else:
            # Combine emotions and numeric columns for correlation
            corr_matrix = data[emotions + numeric_cols].corr()
            # Subset to just show correlations between emotions and other variables
            corr_matrix = corr_matrix.loc[emotions, numeric_cols]
    
    plt.figure(figsize=figsize)
    mask = np.zeros_like(corr_matrix, dtype=bool)
    
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, mask=mask, square=True,
                linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title('Emotion Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_sentiment_distribution(df, sentiment_col='sentiment', by_column=None, figsize=(10, 6), title='Sentiment Distribution'):
    plt.figure(figsize=figsize)
    
    if by_column and by_column in df.columns:
        # Create grouped barplots
        cross_tab = pd.crosstab(df[by_column], df[sentiment_col])
        cross_tab.plot(kind='bar', stacked=True, colormap='viridis')
        plt.xlabel(by_column, fontsize=13)
        plt.ylabel('Count', fontsize=13)
    else:
        # Simple countplot
        value_counts = df[sentiment_col].value_counts().sort_index()
        ax = sns.barplot(x=value_counts.index, y=value_counts.values, palette='viridis')
        plt.xlabel('Sentiment', fontsize=13)
        plt.ylabel('Count', fontsize=13)
        
        # Add count labels
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=11, color='black', xytext=(0, 8), 
                        textcoords='offset points')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_sentiment_emotion_heatmap(df, emotions=None, figsize=(10, 8), cmap='coolwarm', title='Sentiment-Emotion Correlation'):
    if emotions is None:
        emotions = ['anxiety', 'stress', 'confusion', 'hopeful', 'fear']
    
    # Create a correlation matrix
    corr_data = pd.DataFrame()
    
    # Convert sentiment to numerical: negative=0, positive=1
    corr_data['sentiment_score'] = df['sentiment'].map({'negative': 0, 'positive': 1})
    
    # Add emotion columns
    for emotion in emotions:
        if emotion in df.columns:
            corr_data[emotion] = df[emotion]
    
    # Calculate correlation matrix
    corr_matrix = corr_data.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, linewidths=0.5,
                cbar_kws={"shrink": .8})
    
    # Update axis labels to show original sentiment label
    sentiment_label = 'Sentiment\n(0=negative, 1=positive)'
    corr_matrix.index = [sentiment_label if x == 'sentiment_score' else x for x in corr_matrix.index]
    corr_matrix.columns = [sentiment_label if x == 'sentiment_score' else x for x in corr_matrix.columns]
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_bert_training_progress(training_stats, figsize=(12, 5)):
    # Extract metrics
    stats_df = pd.DataFrame(training_stats)
    
    # Create a two-panel plot: training loss and validation metrics
    plt.figure(figsize=figsize)
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(stats_df['epoch'], stats_df['training_loss'], marker='o', linestyle='-', color='blue')
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot validation metrics if available
    if 'val_accuracy' in stats_df.columns:
        plt.subplot(1, 2, 2)
        plt.plot(stats_df['epoch'], stats_df['val_accuracy'], marker='o', linestyle='-', label='Accuracy', color='green')
        
        if 'val_f1' in stats_df.columns:
            plt.plot(stats_df['epoch'], stats_df['val_f1'], marker='s', linestyle='-', label='F1 Score', color='orange')
        
        plt.title('Validation Metrics', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def plot_emotion_by_sentiment(df, emotions=None, figsize=(12, 8), title='Emotions by Sentiment'):
    if emotions is None:
        emotions = ['anxiety', 'stress', 'confusion', 'hopeful', 'fear']
    
    plt.figure(figsize=figsize)
    
    # Group by sentiment and calculate emotion means
    grouped = df.groupby('sentiment_label')[emotions].mean()
    
    # Plot as a heatmap
    sns.heatmap(grouped, annot=True, fmt=".2f", cmap='viridis', linewidths=0.5)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Emotion', fontsize=13)
    plt.ylabel('Sentiment', fontsize=13)
    plt.tight_layout()
    plt.show()

def plot_sentiment_confidence(df, figsize=(10, 6), title='Sentiment Prediction Confidence'):
    if 'confidence' not in df.columns:
        print("No confidence scores available to plot")
        return
    
    plt.figure(figsize=figsize)
    
    # Group by sentiment label
    grouped = df.groupby('sentiment_label')['confidence'].mean().reset_index()
    
    # Plot as a bar chart
    sns.barplot(x='sentiment_label', y='confidence', data=grouped, palette='viridis')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Sentiment', fontsize=13)
    plt.ylabel('Average Confidence Score', fontsize=13)
    plt.ylim(0, 1)  # Confidence scores should be between 0 and 1
    plt.tight_layout()
    plt.show()