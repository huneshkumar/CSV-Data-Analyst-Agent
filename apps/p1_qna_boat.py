import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

# Initialize the model
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.6,
    max_retries=2
)

# Streamlit page config
st.set_page_config(
    page_title="🧠 AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Modern AI Chatbot")
st.markdown("Ask any question and get concise answers from LLaMA 3.1!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# User input
if prompt := st.chat_input("Type your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Prepare LangChain prompt
    chat_prompt = ChatPromptTemplate.from_messages([
        {"role": "system", "content": "You are a helpful assistant. Answer the user question in a concise manner."},
        {"role": "user", "content": "{question}"}
    ])
    chain = chat_prompt | model

    # Get response
    response = chain.invoke({"question": prompt})

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response.content})
    st.chat_message("assistant").write(response.content)