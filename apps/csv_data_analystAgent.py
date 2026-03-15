import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
load_dotenv()  # Load environment variables from .env file
st.title("CSV Data Analyst Agent")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")
    st.dataframe(df.head())

    question = st.text_input("Ask question about dataset")

    if question:

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.6
        )

        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True
        )

        response = agent.invoke(question)

        st.write(response["output"])