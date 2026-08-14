# test_groq.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

try:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ['GROQ_API_KEY']
    )
    print("Testing Groq...")
    resp = llm.invoke("Say 'Hello'")
    print(f"SUCCESS: {resp.content}")
except Exception as e:
    print(f"FAILED: {e}")
