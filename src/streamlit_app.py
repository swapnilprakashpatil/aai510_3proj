import streamlit as st
from src.config import Config
from src.preprocessing import preprocess_data
from src.supervised import predict_show_no_show
from src.unsupervised import cluster_patients
from src.nlp_pipeline import analyze_sentiment, analyze_no_show_reason

def main():
    st.title("Patient Appointments Show/No Show Prediction and Analysis")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = ["Show/No Show Prediction", "Patient Clustering", "Patient Sentiment Analysis", "No-show Reason Analysis"]
    choice = st.sidebar.selectbox("Select an option", options)

    if choice == "Show/No Show Prediction":
        st.subheader("Predict Show/No Show")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            data = preprocess_data(uploaded_file)
            prediction = predict_show_no_show(data)
            st.write("Predictions:", prediction)

    elif choice == "Patient Clustering":
        st.subheader("Patient Clustering")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            data = preprocess_data(uploaded_file)
            clusters = cluster_patients(data)
            st.write("Clusters:", clusters)

    elif choice == "Patient Sentiment Analysis":
        st.subheader("Analyze Patient Sentiment")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            data = preprocess_data(uploaded_file)
            sentiment_analysis = analyze_sentiment(data)
            st.write("Sentiment Analysis Results:", sentiment_analysis)

    elif choice == "No-show Reason Analysis":
        st.subheader("Analyze No-show Reasons")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            data = preprocess_data(uploaded_file)
            no_show_analysis = analyze_no_show_reason(data)
            st.write("No-show Reason Analysis Results:", no_show_analysis)

if __name__ == "__main__":
    main()