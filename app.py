import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Storyteller",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for a Slick Design ---
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    /* Style for the card-like container */
    .stApp > header {
        background-color: transparent;
    }
    .stApp {
        background-color: #f0f2f6; /* Light gray background */
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 25px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    .card:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    /* Button styling */
    .stButton>button {
        color: white;
        background-color: #4CAF50; /* Green */
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)


# --- Caching the Model and Tokenizer for Performance ---
@st.cache_resource
def load_assets():
    """Loads the trained model and tokenizer from disk."""
    model = load_model('next_word_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

model, tokenizer = load_assets()
# Ensure sequence_length is correctly retrieved from the model's input layer
sequence_length = model.input_shape[1]


# --- Text Generation Function ---
def generate_text(seed_text, next_words, model, tokenizer, max_sequence_len):
    """Generates text using the trained LSTM model."""
    output_text = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([output_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_index = np.argmax(predicted_probs)
        
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        
        output_text += " " + output_word
        
    return output_text.title() # Using .title() for a story-like capitalization

# --- App Layout and UI ---

# Header
st.title("✍️ AI Storyteller")
st.markdown("Your creative partner for writing. Powered by an LSTM model trained on *Alice's Adventures in Wonderland*.")
st.divider()

# Main content within a styled container
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.subheader("Start Your Story...")
    
    # Input fields
    seed_text = st.text_area(
        "Enter a starting phrase:", 
        "The Duchess took her choice, and was gone in a moment",
        height=100
    )
    
    num_words = st.slider(
        "How many words would you like to generate?",
        min_value=10, max_value=200, value=50, step=10
    )
    
    # Generate button and output
    if st.button("Generate Text"):
        if seed_text:
            with st.spinner("The AI is thinking..."):
                generated_sentence = generate_text(seed_text, num_words, model, tokenizer, sequence_length)
                st.divider()
                st.subheader("📜 Here is your generated story:")
                st.success(generated_sentence)
        else:
            st.warning("Please enter a starting phrase to begin.")
            
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and TensorFlow.")