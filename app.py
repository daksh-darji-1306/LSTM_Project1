import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import plotly.graph_objects as go
import os

# --- Configuration ---
MODEL_FILE = 'customer_journey_lstm.h5'
ENCODER_FILE = 'event_encoder.pkl'
SEQUENCE_LENGTH = 10

# --- Asset Loading ---
@st.cache_resource
def load_prediction_assets():
    """Loads and validates the trained model and label encoder."""
    if not all(os.path.exists(f) for f in [MODEL_FILE, ENCODER_FILE]):
        return None, None
    try:
        model = load_model(MODEL_FILE)
        encoder = joblib.load(ENCODER_FILE)
        return model, encoder
    except Exception as e:
        st.error(f"Error loading prediction assets: {e}")
        return None, None

# --- UI Setup ---
st.set_page_config(page_title="Customer Journey Predictor", layout="wide", initial_sidebar_state="collapsed")
model, encoder = load_prediction_assets()

# --- Professional Dark Theme CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .main {
        background-color: #0d1117;
    }
    .card-container {
        background-color: #161b22;
        border-radius: 10px;
        padding: 2rem;
        border: 1px solid #30363d;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: 1px solid #30363d;
        padding: 12px 0;
        width: 100%;
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        border-color: #8b949e;
    }
    h1, h2, h3 { color: #f0f6fc !important; }
    .stMarkdown, p, .stMultiSelect { color: #c9d1d9; }
    .prediction-card {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-left: 5px solid #2ea043;
        border-radius: 10px;
        padding: 1.5rem 2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Main Application Logic ---
if model is not None and encoder is not None:
    st.title("🛒 E-commerce Customer Journey Predictor")
    st.markdown("<p style='font-size: 1.1rem; color: #8b949e;'>Select a sequence of customer actions to predict the most likely next event.</p>", unsafe_allow_html=True)
    st.markdown("---")

    with st.container():
        st.markdown("<div class='card-container'>", unsafe_allow_html=True)
        st.header("Customer's Current Journey")
        
        event_options = list(encoder.classes_)
        
        # User input for the sequence of events
        journey_sequence = st.multiselect(
            "Select the sequence of events in chronological order.",
            options=event_options,
            default=[event_options[0], event_options[0]] # Default to two 'view' events
        )
        
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        _, col_btn, _ = st.columns([2.2, 1, 2.2])
        predict_button = col_btn.button("Predict Next Action")

    if predict_button and journey_sequence:
        st.markdown("---")
        st.header("Prediction Result")

        # Prepare the input for the model
        encoded_sequence = encoder.transform(journey_sequence)
        padded_sequence = pad_sequences([encoded_sequence], maxlen=SEQUENCE_LENGTH - 1, padding='pre')

        # Make prediction
        prediction_probabilities = model.predict(padded_sequence)[0]
        predicted_class_index = np.argmax(prediction_probabilities)
        predicted_event = encoder.inverse_transform([predicted_class_index])[0]
        
        with st.container():
            st.markdown("<div class='card-container'>", unsafe_allow_html=True)
            res_cols = st.columns([1, 1.5], gap="large")
            
            with res_cols[0]:
                st.markdown(f"""
                <div class="prediction-card">
                    <p style="color: #8b949e; font-size: 1.1rem; font-weight: 600;">PREDICTED NEXT ACTION</p>
                    <h2 style="color: #f0f6fc; font-size: 2.5rem; font-weight: 700; text-transform: uppercase; letter-spacing: 2px;">{predicted_event}</h2>
                </div>
                """, unsafe_allow_html=True)

            with res_cols[1]:
                # Create a bar chart for probabilities
                fig = go.Figure([go.Bar(
                    x=list(encoder.classes_), 
                    y=prediction_probabilities,
                    marker_color='#238636'
                )])
                fig.update_layout(
                    title_text='Prediction Probabilities',
                    xaxis_title="Event Type",
                    yaxis_title="Probability",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#c9d1d9",
                    title_font_color="#f0f6fc"
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)
            
    elif predict_button and not journey_sequence:
        st.warning("Please select at least one event in the customer journey.")

else:
    st.error("Model assets not found. Please run `train.py` first to generate the required model and encoder files.")
