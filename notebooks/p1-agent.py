from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
load_dotenv()  # Load environment variables from .env file


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.
    Args:
        a (int): The first number.
        b (int): The second number.
    
    """
    return a + b


model = ChatGroq(
    model="llama-3.1-8b-instant",       
    temperature=0.6
)

agent = create_agent(model=model, tools=[add],system_prompt='You are a helpful assistant that can perform addition using the add tool.')

  
response = agent.invoke(
    {
        'messages':[{'role':"user",'content':'what is 2 + 3 ?'}]
    }
)
print(response['messages'])
