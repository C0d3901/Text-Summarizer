# Step 1: Import required libraries
from transformers import pipeline  # For using pre-trained NLP models
import streamlit as st            # For creating a simple web interface

# Step 2: Load the pre-trained summarization model
# Using 'facebook/bart-large-cnn', which is a powerful summarization model
# This may take a few seconds the first time you run it, as it downloads the model
@st.cache_resource  # Cache the model to prevent reloading on every run
def load_model():
    summarizer_model = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer_model

model = load_model()

# Step 3: Create the Streamlit UI
st.set_page_config(page_title="Text Summarizer", page_icon="📝", layout="centered")

# Title and description
st.title("📝 Text Summarizer")
st.write("""
Paste any article, paragraph, or text in the box below, 
and this app will automatically generate a concise summary using HuggingFace BART.
""")

# Step 4: Add a text input area
user_text = st.text_area(
    "Enter text here:",
    height=250,
    placeholder="Type or paste your text here..."
)

# Step 5: Create a Summarize button
if st.button("Summarize"):
    
    # Check if the user entered any text
    if user_text.strip() != "":
        # Step 5a: Summarize the input text
        # Parameters:
        # max_length: maximum number of tokens in the summary
        # min_length: minimum number of tokens in the summary
        # do_sample=False: ensures deterministic output
        summary_result = model(user_text, max_length=130, min_length=30, do_sample=False)
        
        # Step 5b: Display the summary
        st.subheader("Summary:")
        st.write(summary_result[0]['summary_text'])
    
    else:
        # If no text is entered, display a message
        st.warning("⚠️ Please enter some text to summarize!")

# Step 6: Optional: Add a footer
st.markdown("---")
st.markdown("Made with ❤️ using Python, HuggingFace Transformers, and Streamlit.")

