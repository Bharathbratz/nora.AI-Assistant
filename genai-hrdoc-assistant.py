# HRDoc AI Assistant
# Streamlit + LangChain + OpenAI + FAISS

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import tempfile
import os

# Sidebar config
st.set_page_config(page_title="HRDoc AI Assistant", layout="centered")
st.title("📄 HRDoc AI Assistant")
st.markdown("Ask your HR policy questions based on uploaded PDFs.")

# OpenAI API key
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

uploaded_file = st.file_uploader("Upload HR Policy PDF", type="pdf")
query = st.text_input("Enter your question")

if uploaded_file and openai_api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Load and split PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Embedding and FAISS
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    # QA Chain
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=openai_api_key, temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    if query:
        with st.spinner("Generating answer..."):
            result = qa({"query": query})
            st.success(result["result"])

            # Source docs
            with st.expander("Source Documents"):
                for doc in result["source_documents"]:
                    st.markdown(f"- {doc.metadata['source']}")

    # Cleanup
    os.remove(tmp_path)

elif not openai_api_key:
    st.warning("🔑 Please enter your OpenAI API key.")

