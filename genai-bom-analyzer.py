# Engineering BOM Analyzer Agent
# Streamlit + CSV + LangChain + OpenAI

import streamlit as st
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import tempfile

st.set_page_config(page_title="⚙️ BOM Analyzer Agent", layout="centered")
st.title("⚙️ Engineering BOM Analyzer")
st.markdown("Upload a BOM CSV file and ask compatibility or part-related questions.")

openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
uploaded_file = st.file_uploader("Upload BOM CSV File", type="csv")
query = st.text_input("Ask a question about the BOM (e.g., compatibility of Part X and Y)")

if uploaded_file and openai_api_key:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 BOM Preview")
    st.dataframe(df.head())

    if query:
        with st.spinner("Analyzing your BOM query..."):
            llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
            agent = create_pandas_dataframe_agent(llm, df, verbose=False)
            response = agent.run(query)
        st.success("✅ Response")
        st.write(response)

elif not openai_api_key:
    st.warning("🔑 Please enter your OpenAI API key.")
