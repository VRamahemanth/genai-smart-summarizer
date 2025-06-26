import streamlit as st
from transformers import pipeline
from pdfminer.high_level import extract_text
import io

st.set_page_config(page_title="GenAI Research Summarizer Hemanth")
st.title("📄 Smart Assistant: Upload & Summarize Your Document by Hemanth")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
    file_text = ""

    if uploaded_file.type == "application/pdf":
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        file_text = extract_text("temp.pdf")

    elif uploaded_file.type == "text/plain":
        file_text = uploaded_file.read().decode("utf-8")

    st.success("✅ Document uploaded and processed successfully!")

    with st.expander("🔍 View Extracted Text"):
        st.write(file_text[:1000] + "..." if len(file_text) > 1000 else file_text)

    with st.spinner("Generating summary..."):
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summary = summarizer(file_text, max_length=150, min_length=50, do_sample=False)
        st.subheader("📝 Auto-Generated Summary:")
        st.write(summary[0]['summary_text'])
