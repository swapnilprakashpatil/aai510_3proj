import streamlit as st
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import time
# Internal imports
from src.clinical_notes_prediction import ClinicalNotesNoShowPredictor
from src.config import PREDICTION_MODEL_EXPORT_PATH
import warnings

warnings.filterwarnings("ignore")

# Add a patient-doctor emoji at the top for visual appeal
st.markdown("""
<div style='text-align: center;'>
    <span style='font-size: 60px;'>🧑‍⚕️🧑‍🦽</span>
</div>
""", unsafe_allow_html=True)

st.title("Patient Sentiment Emotion Analyzer")

# Sidebar: Model selection (for now, only TinyBERT)
st.sidebar.title("Model Selection")
model_options = [
    {
        "text": "Sentiment Analysis (TinyBERT)",
        "value": "tinybert",
        "placeholder": "Example: Patient expresses anxiety about upcoming procedure and confusion regarding medication instructions.",
        "button_text": "Analyze Patient Sentiment",
        "examples": [
            "Patient is hopeful and shows no significant anxiety, stress, or fear related to health conditions.",
            "Patient expresses fear and anxiety about high blood pressure and possible complications.",
            "Elderly patient expresses fear of declining health, confusion about medications, and stress related to mobility issues.",
            "Patient (minor) is anxious and fearful about medical procedures, sometimes confused by instructions, and stressed by separation from family.",
            "Patient expresses confusion about medication schedule and is hopeful about recovery."
        ]
    },
    {
        "text": "Clinical Notes Analysis (TinyClinicalBERT)",
        "value": "clinicalbert",
        "placeholder": "Example: Patient missed previous appointments due to transportation issues.",
        "button_text": "Predict No-Show from Clinical Note",
        "examples": [
            "Patient reported forgetting the appointment.",
            "Missed due to lack of transportation.",
            "Appointment was too early in the morning.",
            "Patient was not feeling well and decided to cancel last minute.",
            "Patient will attend the appointment as scheduled."
        ]
    }
]
if 'selected_model_index' not in st.session_state:
    st.session_state.selected_model_index = 0
if 'text_input' not in st.session_state:
    st.session_state.text_input = ""
selected_model_dict = st.sidebar.selectbox(
    "Choose a model:",
    model_options,
    format_func=lambda x: x["text"],
    index=st.session_state.selected_model_index,
    key="model_selectbox"
)
if st.session_state.selected_model_index != model_options.index(selected_model_dict):
    with st.spinner("Loading model, please wait..."):
        progress_bar = st.progress(0)
        for percent in range(0, 101, 10):
            time.sleep(0.03)
            progress_bar.progress(percent)
        progress_bar.empty()  # Remove Streamlit bar
        # Custom blue and smaller bar (height: 18px)
        st.markdown("""
        <div style='width: 100%; background: #e0e0e0; border-radius: 10px; height: 18px; margin: 10px 0;'>
            <div style='width: 100%; background: #2196F3; height: 18px; border-radius: 10px;'></div>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.selected_model_index = model_options.index(selected_model_dict)
        st.session_state.text_input = ""  # Clear textbox
        st.rerun()
selected_model = selected_model_dict["value"]

# Example patient sentiment notes (from app.py)
examples = selected_model_dict["examples"]

# Text input area with example buttons
st.subheader(f"Enter {selected_model_dict['text']}")
input_text = st.text_area(
    f"Type or paste input here:",
    value=st.session_state.text_input,
    height=150,
    placeholder=selected_model_dict["placeholder"]
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

if selected_model == "clinicalbert":
    @st.cache_resource
    def load_clinicalbert_predictor():
        model_dir = PREDICTION_MODEL_EXPORT_PATH
        predictor = ClinicalNotesNoShowPredictor()
        predictor.model = predictor.model.from_pretrained(model_dir)
        predictor.tokenizer = predictor.tokenizer.from_pretrained(model_dir)
        return predictor
    clinicalbert_predictor = load_clinicalbert_predictor()

    def predict_noshow(text):
        preds = clinicalbert_predictor.predict([text])
        return int(preds[0])

# Add emoji variables for no-show and calendar
NOSHOW_EMOJI = "\U0001F6AB"  # 🚫
CALENDAR_EMOJI = "\U0001F4C5"  # 📅

# Example buttons (use examples from selected model)
col1, col2 = st.columns(2)
with col1:
    if st.button("Example 1"):
        st.session_state.text_input = examples[0]
        st.rerun()
    if len(examples) > 2 and st.button("Example 3"):
        st.session_state.text_input = examples[2]
        st.rerun()
with col2:
    if len(examples) > 1 and st.button("Example 2"):
        st.session_state.text_input = examples[1]
        st.rerun()
    if len(examples) > 3 and st.button("Example 4"):
        st.session_state.text_input = examples[3]
        st.rerun()

# Analyse button
def show_results():
    if input_text:
        if selected_model == "tinybert":
            model, tokenizer, device = load_tinybert_model()
            with st.spinner("Analyzing emotions..."):
                results, probs = predict_emotions_raw(input_text, model, tokenizer, device)
            st.subheader("Detected Emotions")
            detected = [
                f"<div style='display: inline-block; text-align: center; margin: 0 24px;'><div style='font-size: 6em'>{emotion_emojis[emo]}</div><div style='font-size: 1.2em; margin-top: 0.2em'>{emo.capitalize()}</div></div>"
                for emo, present in results.items() if present
            ]
            if detected:
                html = "<div style='display: flex; justify-content: center;'>" + "".join(detected) + "</div>"
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.info("No significant emotions detected.")
        else:
            with st.spinner("Analyzing clinical note..."):
                noshow_pred = predict_noshow(input_text)
            st.subheader("Clinical Note No-Show Prediction")
            if noshow_pred == 1:
                st.success(f"Predicted: No-show {NOSHOW_EMOJI}")
            else:
                st.info(f"Predicted: Show {CALENDAR_EMOJI}")
    else:
        st.warning("Please enter some text to analyze.")

# Update button logic to show progress bar
if st.button(selected_model_dict["button_text"], type="primary"):
    progress_bar = st.progress(0)
    for percent in range(0, 101, 10):
        time.sleep(0.01)
        progress_bar.progress(percent)
    progress_bar.empty()  # Remove Streamlit bar
    st.markdown("""
    <div style='width: 100%; background: #e0e0e0; border-radius: 10px; height: 18px; margin: 10px 0;'>
        <div style='width: 100%; background: #2196F3; height: 18px; border-radius: 10px;'></div>
    </div>
    """, unsafe_allow_html=True)
    show_results()
