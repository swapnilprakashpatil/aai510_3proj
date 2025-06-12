import streamlit as st
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Add a patient-doctor emoji at the top for visual appeal
st.markdown("""
<div style='text-align: center;'>
    <span style='font-size: 60px;'>🧑‍⚕️🧑‍🦽</span>
</div>
""", unsafe_allow_html=True)

st.title("Patient Sentiment Emotion Analyzer")

# Sidebar: Model selection (for now, only TinyBERT)
st.sidebar.title("Model Selection")
model_options = ["Sentiment Analysis (TinyBERT)"]
selected_model = st.sidebar.selectbox("Choose a model:", model_options)

# Example patient sentiment notes (from app.py)
examples = [
    "Patient is hopeful and shows no significant anxiety, stress, or fear related to health conditions.",
    "Patient expresses fear and anxiety about high blood pressure and possible complications.",
    "Elderly patient expresses fear of declining health, confusion about medications, and stress related to mobility issues.",
    "Patient (minor) is anxious and fearful about medical procedures, sometimes confused by instructions, and stressed by separation from family.",
    "Patient expresses confusion about medication schedule and is hopeful about recovery."
]

# Text input area with example buttons
st.subheader("Enter Patient Sentiment Notes")
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""

col1, col2 = st.columns(2)
with col1:
    if st.button("Example 1"):
        st.session_state.text_input = examples[0]
    if st.button("Example 3"):
        st.session_state.text_input = examples[2]
with col2:
    if st.button("Example 2"):
        st.session_state.text_input = examples[1]
    if st.button("Example 4"):
        st.session_state.text_input = examples[3]

input_text = st.text_area(
    "Type or paste patient's sentiment here:",
    value=st.session_state.text_input,
    height=150,
    placeholder="Example: Patient expresses anxiety about upcoming procedure and confusion regarding medication instructions."
)

# Load model and tokenizer (TinyBERT)
@st.cache_resource
def load_tinybert_model():
    model_dir = os.path.join("models", "nlp", "sentiment_analysis_optimized")  # Use optimized model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model = model.to(device)
    return model, tokenizer, device

# Emotion labels and emojis
emotion_emojis = {
    'anxiety': '😰',
    'stress': '😓',
    'confusion': '😕',
    'hopeful': '🤞',
    'fear': '😨'
}
emotion_labels = list(emotion_emojis.keys())

# Predict emotions (raw, no post-processing)
def predict_emotions_raw(text, model, tokenizer, device):
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
    # Use 0.5 threshold for all emotions
    results = {emo: bool(probs[i] >= 0.5) for i, emo in enumerate(emotion_labels)}
    return results, probs

# Analyse button
def show_results():
    if input_text:
        model, tokenizer, device = load_tinybert_model()
        with st.spinner("Analyzing emotions..."):
            results, probs = predict_emotions_raw(input_text, model, tokenizer, device)
        st.subheader("Detected Emotions")
        detected = [f"{emotion_emojis[emo]} {emo.capitalize()}" for emo, present in results.items() if present]
        if detected:
            st.success(f"Detected: {', '.join(detected)}")
        else:
            st.info("No significant emotions detected.")
        # Show bar chart of probabilities
        chart_data = pd.DataFrame({
            'Emotion': [f"{emotion_emojis[emo]} {emo.capitalize()}" for emo in emotion_labels],
            'Probability': probs
        })
        chart_data = chart_data.sort_values('Probability', ascending=False)
        st.bar_chart(chart_data.set_index('Emotion'))
    else:
        st.warning("Please enter some text to analyze.")

if st.button("Analyze Patient Sentiment", type="primary"):
    show_results()
