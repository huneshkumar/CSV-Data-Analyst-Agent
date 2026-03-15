import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

load_dotenv()

st.set_page_config(page_title="CSV AI Analyst", page_icon="📊")

st.title("📊 AI CSV Data Analyst")

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    # Load dataframe once
    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(uploaded_file)

    df = st.session_state.df

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Create LLM + Agent only once
    # -----------------------------
    if "agent" not in st.session_state:

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0
        )

        st.session_state.agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True
        )

    agent = st.session_state.agent

    # -----------------------------
    # Chat History
    # -----------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # -----------------------------
    # Chat Input
    # -----------------------------
    prompt = st.chat_input("Ask something about the dataset...")

    if prompt:

        # Show user message
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # -----------------------------
        # Loader / Spinner
        # -----------------------------
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data... 📊"):

                response = agent.invoke(prompt)
                answer = response["output"]

                st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })