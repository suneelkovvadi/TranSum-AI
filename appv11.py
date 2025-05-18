# Import Libraries
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline, MarianMTModel, MarianTokenizer
import torch
import os
import re

#Load T5 Model and Tokenizer
# Load once
checkpoint = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint)

#Translator Function with Caching
@st.cache_resource
def get_translator(src_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    def translate(text):
        batch = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt", truncation=True)
        translated = model.generate(**batch)
        return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return translate

#Fix Capitalization
def fix_capitalization(text):
    return re.sub(r'(?<=[.!?])\s+(.)', lambda m: ' ' + m.group(1).upper(), text.capitalize())

# PDF Preprocessing
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = splitter.split_documents(pages)
    return " ".join([t.page_content for t in texts])

#Summarization Pipeline Loader
@st.cache_resource
def get_pipeline(level):
    config = {
        "Short": {"max_length": 100, "min_length": 30},
        "Medium": {"max_length": 250, "min_length": 50},
        "Long": {"max_length": 500, "min_length": 80}
    }
    return pipeline("summarization", model=base_model, tokenizer=tokenizer, **config[level])

#Summarization Function
def summarize_text(text, level):
    pipe = get_pipeline(level)
    return fix_capitalization(pipe(text)[0]['summary_text'])

#Streamlit UI Configuration
# Streamlit UI
st.set_page_config(page_title="TranSum AI", layout="wide")

#Main Function with UI
def main():
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>📄 TranSum AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size:18px;'>Translate and Summarize PDF documents with AI</p>", unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.markdown("### 🌐 Language Selection")
        lang_dict = {
            "English": "en", "French": "fr", "German": "de", "Spanish": "es",
            "Hindi": "hi", "Chinese": "zh", "Arabic": "ar", "Russian": "ru",
            "Japanese": "ja", "Italian": "it", "Portuguese": "pt", "Polish": "pl",
            "Dutch": "nl", "Turkish": "tr"
        }
        lang_name = st.selectbox("Choose the input language of your PDF:", list(lang_dict.keys()))
        lang_code = lang_dict[lang_name]

        st.markdown("### 📤 Upload PDF")
        uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

        if st.button("🚀 Extract Text") and uploaded_file:
            filepath = "data/" + uploaded_file.name
            os.makedirs("data", exist_ok=True)
            with open(filepath, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

            raw_text = file_preprocessing(filepath)
            st.session_state["raw_text"] = raw_text

            if lang_code != "en":
                translator = get_translator(lang_code)
                st.session_state["final_text"] = translator(raw_text)
            else:
                st.session_state["final_text"] = raw_text

    if "raw_text" in st.session_state:
        st.markdown("#### 📘 Extracted Text in Original Language")
        st.text_area("Extracted Text", st.session_state["raw_text"], height=200)

    if "final_text" in st.session_state and lang_code != "en":
        st.markdown("#### 🌐 Translated Text")
        st.text_area("Translated Text", st.session_state["final_text"], height=200)

    if "final_text" in st.session_state:
        level = st.radio("✂️ Choose Summary Level:", ("Short", "Medium", "Long"))
        if st.button("📌 Generate Summary"):
            summary = summarize_text(st.session_state["final_text"], level)
            st.markdown("#### 📋 Summarized Text")
            st.success(summary)

if __name__ == "__main__":
    main()
