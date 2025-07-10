import streamlit as st
import torch
import fitz
from transformers import LEDTokenizer, LEDForConditionalGeneration
import re

@st.cache_resource
def load_model():
    model_name = 'allenai/led-large-16384-arxiv'
    try:
        tokenizer = LEDTokenizer.from_pretrained(model_name)
        model = LEDForConditionalGeneration.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load the model {e}")
        return None, None

def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    try:
        if not uploaded_file.name.endswith('.pdf'):
            st.error("Please upload a valid PDF file.")
            return None

        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        return full_text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def summarize_text(text, model, tokenizer):
    """Generates a summary for the given text using the loaded model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(text, max_length=1024, return_tensors='pt', truncation=True).to(device)

    summary_ids = model.generate(
        inputs['input_ids'], 
        num_beams=4, 
        max_length=500,  
        min_length=100,
        early_stopping=True
    )

    # Decode the summary
    summary = tokenizer.decode(
        summary_ids[0], 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return summary

def post_process_summary(text):
    text=text.strip()
    if not text:
        return ""   
    sentences=re.split(r'(?<=[.!?])\s*', text)
    processed_sentences=[]
    for sentence in sentences:
        s=sentence.strip()
        if s:
            processed_sentences.append(s[0].upper()+s[1:]) 
    final_summary=" ".join(processed_sentences)
    if final_summary and final_summary[-1] not in ['.', '!', '?']:
        final_summary += '.'
        
    return final_summary



# --- Streamlit App Interface ---
st.set_page_config(page_title="Summary", layout="wide")

st.title(" Summarizer ")
st.markdown("Paste a paragraph or upload a PDF and this app will generate a concise summary using a sophisticated AI model.")

# Load the model and tokenizer
model, tokenizer = load_model()

# --- Corrected Code Logic ---

if model is not None and tokenizer is not None:
    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"]) 
    
    # This outer 'if' handles the entire workflow for an uploaded file
    if uploaded_file is not None:
        st.info("Processing your PDF file...")
        with st.spinner("Extracting text from PDF..."):
            # Attempt to extract text
            pdf_text = extract_text_from_pdf(uploaded_file)

        # --- KEY CHANGE: The rest of the logic is now nested inside this check ---
        if pdf_text and pdf_text.strip():
            st.success("Text extracted successfully!") 
            st.text_area("Extracted Text", value=pdf_text, height=200, disabled=True)

            # The button only appears and is processed AFTER text has been confirmed
            if st.button("Generate Summary", key="pdf_summary_button"):

                with st.spinner("Thinking..."):
                    raw_summary = summarize_text(pdf_text, model, tokenizer)
                    final_summary = post_process_summary(raw_summary)

                st.subheader("Generated Summary")
                st.success(final_summary)
        
        else:
            # This 'else' catches failures in text extraction
            st.error("Failed to extract text from the PDF. It might be an image-based file or corrupted.") 
    
    else:
        # This 'else' shows a message when the page first loads
        st.info("Please upload a PDF file to begin.")

    st.subheader("Input Paragraph")
    # Create a text input for the user to paste their paragraph
    paragraph_input = st.text_area("Paste your paragraph here:", height=200, placeholder="Type or paste your paragraph here...")
    
    if st.button("Generate Summary", key="paragraph_summary_button"): 
        if paragraph_input and paragraph_input.strip():
            with st.spinner("Thinking"):

                raw_summary = summarize_text(paragraph_input, model, tokenizer)
                final_summary=post_process_summary(raw_summary)

            st.subheader("Generated Summary")
            st.success(final_summary)
        else:
            st.warning("Please paste a paragraph to summarize.")
else:
    st.error("Model could not be loaded. The app cannot function without it.")