import streamlit as st
import torch
import numpy as np
import pandas as pd
import os
import time
import io
import warnings
import matplotlib
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from medspacy.visualization import visualize_ent
from wordcloud import WordCloud

# Internal imports
import src.config as config
from src.config import NLP_CONFIG, PREDICTION_MODEL_EXPORT_PATH, SENTIMENT_MODEL_EXPORT_PATH_RAW, TOPIC_MODEL_EXPORT_PATH
from src.sentiment_analysis import SentimentAnalysisModel
from src.sentiment_analysis import EmotionPostProcessor
from src.clinical_notes_prediction import ClinicalNotesNoShowPredictor
from src.clinical_topic_model import ClinicalTopicModel

matplotlib.use('Agg')

warnings.filterwarnings("ignore")

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
    },
    {
        "text": "Topic Modeling (LDA) - Clinical Notes",
        "value": "topicmodel",
        "placeholder": "Patient with diabetes is receiving alcohol use screening. The patient is managing type 2 diabetes with Metformin 500mg twice daily. We analyzed home glucose monitoring data and reinforced the significance of dietary discipline and physical activity. A follow-up for HbA1c testing was arranged. Patient with type 2 diabetes presented with recurrent hypoglycemic episodes (blood glucose <60 mg/dL). Adjusted insulin regimen and provided education on recognizing early symptoms.",
        "button_text": "Analyze Topics from Clinical Note",
        "examples": [
            "Patient with diabetes is receiving alcohol use screening. The patient is managing type 2 diabetes with Metformin 500mg twice daily. We analyzed home glucose monitoring data and reinforced the significance of dietary discipline and physical activity. A follow-up for HbA1c testing was arranged. Patient with type 2 diabetes presented with recurrent hypoglycemic episodes (blood glucose <60 mg/dL). Adjusted insulin regimen and provided education on recognizing early symptoms.",
            "Patient continues to struggle with alcohol use disorder. Outlined a stepwise plan for reduction and offered referral to addiction specialist. Patient with alcoholism ( fasting glucose : 128 mg/dL) was started on metformin 500mg BID after abnormal results. Patient with alcoholism is receiving patient education.",
            "Patient with hypertension is receiving diabetes screening. Patient with hypertension (BP: 150/95 mmHg) completed alcohol screening (AUDIT-C: 6). Advised to limit intake and continued hydrochlorothiazide 25mg daily. This patient has a longstanding history of elevated blood pressure. We revisited the importance of medication compliance and healthy lifestyle choices. Educational resources on hypertension management were provided to support self-care."
        ]
    }
]

def load_tinybert_model():
    model_dir = SENTIMENT_MODEL_EXPORT_PATH_RAW
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SentimentAnalysisModel.load_from_pretrained(model_dir, device=NLP_CONFIG['device'])
    tokenizer = model.tokenizer
    return model, tokenizer, device

@st.cache_resource
def load_clinicalbert_predictor():
    model_dir = PREDICTION_MODEL_EXPORT_PATH
    predictor = ClinicalNotesNoShowPredictor()
    predictor.model = predictor.model.from_pretrained(model_dir)
    predictor.tokenizer = predictor.tokenizer.from_pretrained(model_dir)
    return predictor

@st.cache_resource
def load_topic_model():
    lda_model = joblib.load(os.path.join(TOPIC_MODEL_EXPORT_PATH, "lda_model.joblib"))
    vectorizer = joblib.load(os.path.join(TOPIC_MODEL_EXPORT_PATH, "vectorizer.joblib"))
    return lda_model, vectorizer

@st.cache_resource
def get_clinical_topic_model():
    model = ClinicalTopicModel(config)
    return model

# Emotion labels and emojis
emotion_emojis = {
    'anxiety': '😰',
    'stress': '😓',
    'confusion': '😕',
    'hopeful': '🤞',
    'fear': '😨'
}
emotion_labels = list(emotion_emojis.keys())


def predict_emotions(text, model, tokenizer, device):
    from src.config import EMOTION_VARIATIONS_PATH, NEGATION_PATTERNS_PATH

    postprocessor = EmotionPostProcessor(
        emotion_variations_path=EMOTION_VARIATIONS_PATH,
        negation_patterns_path=NEGATION_PATTERNS_PATH
    )
    # Use model.model for both postprocessor and inference
    results = postprocessor.predict(text, model.model, tokenizer, device)
    model.model.eval()
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
        outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    return results, probs

def predict_noshow(text, predictor):
    preds = predictor.predict([text])
    return int(preds[0])

def get_top_topic_words(text, lda_model, vectorizer, n_top_words=10):
    X = vectorizer.transform([text])
    topic_dist = lda_model.transform(X)[0]
    top_topic = topic_dist.argmax()
    feature_names = vectorizer.get_feature_names_out()
    topic_words = [feature_names[i] for i in lda_model.components_[top_topic].argsort()[:-n_top_words - 1:-1]]
    return top_topic, topic_words, topic_dist

def plot_wordcloud_streamlit(words):
    word_freq = {word: 1 for word in words}
    wc = WordCloud(width=800, height=300, background_color='white').generate_from_frequencies(word_freq)
    buf = io.BytesIO()
    wc.to_image().save(buf, format='PNG')
    buf.seek(0)
    st.image(buf, caption='Word Cloud for Top Topic Words', use_container_width=True)

def plot_highlighted_ents(text, clinical_topic_model):
    doc = clinical_topic_model.nlp(text)
    entity_styles = {
        "PROBLEM": {"emoji": "🩺", "bg_color": "#ff0033"},
        "MEDICATION": {"emoji": "💊", "bg_color": "#06bdf5"},
        "PROCEDURE": {"emoji": "🏥", "bg_color": "#7900b1"},
        "TEST": {"emoji": "🧪", "bg_color": "#B94603"},
        "TEST_RESULT": {"emoji": "📊", "bg_color": "#3CBE00"},
        "HISTORY": {"emoji": "📖", "bg_color": "#00f8a5"}
    }
    context_styles = {
        "is_negated": {"label": "NEGATED", "color": "#222", "bg": "#a901b2", "emoji": "🚫"},
        "is_uncertain": {"label": "UNCERTAIN", "color": "#222", "bg": "#a901b2", "emoji": "❓"},
        "is_possible": {"label": "POSSIBLE", "color": "#222", "bg": "#a901b2", "emoji": "🤔"},
        "is_historical": {"label": "HISTORICAL", "color": "#222", "bg": "#a901b2", "emoji": "⏳"},
        "is_hypothetical": {"label": "HYPOTHETICAL", "color": "#222", "bg": "#a901b2", "emoji": "💭"},
        "is_family": {"label": "FAMILY", "color": "#222", "bg": "#a901b2", "emoji": "👪"},
    }
    html = ""
    last_end = 0
    for ent in doc.ents:
        html += text[last_end:ent.start_char]
        style = entity_styles.get(ent.label_, {"emoji": "🔹", "bg_color": "#f0f0f0"})
        context_tags = []
        context_emojis = []
        context_bg = None
        for ctx, ctx_style in context_styles.items():
            if hasattr(ent._, ctx) and getattr(ent._, ctx):
                context_tags.append(ctx_style["emoji"] + " " + ctx_style["label"])
                context_emojis.append(ctx_style["emoji"])
                context_bg = ctx_style["bg"]  # Use the last found context bg
        # Compose context badge
        badge = ""
        if context_tags:
            badge = f'<span style="font-size:1em; margin-right:2px;">{" ".join(context_emojis)}</span>'
        # Use context bg if present, else entity bg
        bg_color = context_bg if context_bg else style["bg_color"]
        html += (
            f'<span style="font-size:1.2em;" title="{ent.label_}">{style["emoji"]}</span>'
            f'{badge}'
            f'<span style="padding:2px 6px; border-radius:4px; background:{bg_color}; margin-right:2px; border:1.5px solid #888;">{ent.text}</span>'
        )
        last_end = ent.end_char
    html += text[last_end:]

    st.markdown(f"### Highlighted Entities")
    st.markdown(f"<div style='font-family:monospace'>{html}</div>", unsafe_allow_html=True)
    # Legend
    legend_html = "<b>Legend:</b> " + " ".join([f'{v["emoji"]}={v["label"]}' for v in context_styles.values()])
    st.markdown(legend_html, unsafe_allow_html=True)
    st.write('Entities:', [(ent.text, ent.label_, {ctx: getattr(ent._, ctx, False) for ctx in context_styles.keys()}) for ent in doc.ents])

def show_tinybert_results(input_text):
    model, tokenizer, device = load_tinybert_model()
    with st.spinner("Analyzing emotions..."):
        results, probs = predict_emotions(input_text, model, tokenizer, device)
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

def show_topicmodel_results(input_text):
    lda_model, vectorizer = load_topic_model()
    clinical_topic_model = get_clinical_topic_model()
    with st.spinner("Analyzing topics from clinical note..."):
        top_topic, topic_words, topic_dist = get_top_topic_words(input_text, lda_model, vectorizer)
    st.write(f"**Top Words:** {', '.join(topic_words)}")
    plot_wordcloud_streamlit(topic_words)
    plot_highlighted_ents(input_text, clinical_topic_model)

def show_clinicalbert_results(input_text):
    clinicalbert_predictor = load_clinicalbert_predictor()
    with st.spinner("Analyzing clinical note..."):
        noshow_pred = predict_noshow(input_text, clinicalbert_predictor)
    st.subheader("Clinical Note No-Show Prediction")
    if noshow_pred == 1:
        st.success(f"Predicted: No-show {NOSHOW_EMOJI}")
    else:
        st.info(f"Predicted: Show {CALENDAR_EMOJI}")

NOSHOW_EMOJI = "🚫"
CALENDAR_EMOJI = "📅"

def main():
    # Sidebar
    st.sidebar.markdown("""
    <div style='text-align: center;'>
        <span style='font-size: 120px;'>🧑‍⚕️</span></br>
        <span style='font-size: 30px;'>SmartCARE.ai</span></br>
        <span style='font-size: 15px;'>UNDERSTANDING NO-SHOWS & PATIENT BEHAVIOR FOR SMART SCHEDULING</span></br></br>
    </div>
    """, unsafe_allow_html=True)

    if 'selected_model_index' not in st.session_state:
        st.session_state.selected_model_index = 0
    if 'text_input' not in st.session_state:
        # Preset diabetes text if topicmodel is selected
        if model_options[st.session_state.selected_model_index]["value"] == "topicmodel":
            st.session_state.text_input = model_options[[m["value"] for m in model_options].index("topicmodel")]["examples"][0]
        else:
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
    if selected_model == "topicmodel" and st.session_state.text_input == "":
        st.session_state.text_input = model_options[[m["value"] for m in model_options].index("topicmodel")]["examples"][0]

    st.subheader(f"Enter {selected_model_dict['text']}")

    # --- BUTTONS FIRST ---
    if selected_model == "topicmodel":
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Diabetes", key="ex1", help="Example: Diabetes", type="secondary", icon="🩸"):
                st.session_state.text_input = examples[0]
                st.rerun()
        with col2:
            if st.button("Hypertension", key="ex2", help="Example: Hypertension", type="secondary", icon="🤕"):
                st.session_state.text_input = examples[2]
                st.rerun()
        with col3:
            if st.button("Alcoholism", key="ex3", help="Example: Alcoholism", type="secondary", icon="🍺"):
                st.session_state.text_input = examples[1]
                st.rerun()
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Example 1", key="ex1", type="secondary", icon="🤞"):
                st.session_state.text_input = examples[0]
                st.rerun()
        with col2:
            if len(examples) > 1 and st.button("Example 2", key="ex2", type="secondary", icon="😰"):
                st.session_state.text_input = examples[1]
                st.rerun()
        with col3:
            if len(examples) > 2 and st.button("Example 3", key="ex3", type="secondary", icon="😰"):
                st.session_state.text_input = examples[2]
                st.rerun()
        with col4:
            if len(examples) > 3 and st.button("Example 4", key="ex4", type="secondary", icon="😕"):
                st.session_state.text_input = examples[3]
                st.rerun()

    # --- TEXTAREA AT THE BOTTOM ---
    input_text = st.text_area(
        f"Type or paste input here:",
        value=st.session_state.text_input,
        height=150,
        placeholder=selected_model_dict["placeholder"]
    )
    st.session_state.text_input = input_text

    if st.button(selected_model_dict["button_text"], key="analyze", type="primary"):
        progress_bar = st.progress(0)
        for percent in range(0, 101, 10):
            time.sleep(0.01)
            progress_bar.progress(percent)
        progress_bar.empty()
        st.markdown("""
        <div style='width: 100%; background: #e0e0e0; border-radius: 10px; height: 18px; margin: 10px 0;'>
            <div style='width: 100%; background: #2196F3; height: 18px; border-radius: 10px;'></div>
        </div>
        """, unsafe_allow_html=True)
        if selected_model == "tinybert":
            show_tinybert_results(input_text)
        elif selected_model == "topicmodel":
            show_topicmodel_results(input_text)
        else:
            show_clinicalbert_results(input_text)

if __name__ == "__main__":
    main()
