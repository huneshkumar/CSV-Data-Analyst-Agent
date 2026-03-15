from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
    # other params...
)

prompt=ChatPromptTemplate.from_messages([
    {
        "role": "system",
        "content": "You are a {role}. Answer the user question in a helpful and concise manner."
    },
    {
        "role": "user",
        "content": "{question}"
    }
])

# final_prompt=prompt.format(role="python developer", question="What is the difference between a list and a tuple in Python?")
# response=model.invoke(final_prompt)

# resP=prompt.invoke({
#     "role": "python developer",
#     "question": "What is the difference between a list and a tuple in Python?"
# })
# response=model.invoke(resP)

#power of lang chain 

chains=prompt | model

response=chains.invoke({
    "role": "python developer",
    "question": "What is the difference between a list and a tuple in Python?"
})

print(response.content)