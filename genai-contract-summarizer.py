# Procurement Contract Summary Bot
# Streamlit + OpenAI + LangChain

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
import tempfile
import os

st.set_page_config(page_title="📜 Contract Summary Bot", layout="centered")
st.title("📜 Procurement Contract Summary Bot")
st.markdown("Upload a vendor contract and get a smart summary in key business terms.")

openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
uploaded_file = st.file_uploader("Upload Contract PDF", type="pdf")

if uploaded_file and openai_api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    llm = OpenAI(temperature=0.3, openai_api_key=openai_api_key)
    chain = load_summarize_chain(llm=llm, chain_type="map_reduce")

    with st.spinner("Summarizing contract..."):
        summary = chain.run(docs)

    st.success("✅ Summary Generated")
    st.subheader("📝 Key Highlights")
    st.markdown(summary)

    os.remove(tmp_path)

elif not openai_api_key:
    st.warning("🔑 Please enter your OpenAI API key.")
