# backend.py

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
# pyrefly: ignore [missing-import]
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
# pyrefly: ignore [missing-import]
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
import sqlite3
# pyrefly: ignore [missing-import]
from langgraph.checkpoint.sqlite import SqliteSaver
import requests
import os
import json
from langchain_core.messages import SystemMessage
from rag_backend import retrieve_context
# pyrefly: ignore [missing-import]
from groq import APIError as GroqAPIError

load_dotenv()

# -------------------
# 1. LLM
# -------------------
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_api_key,
    temperature=0,       # deterministic = more reliable tool-call JSON
    max_retries=2,
)

# -------------------
# 2. Tools
# -------------------
# Tools
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> str:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return json.dumps({"error": "Division by zero is not allowed"})
            result = first_num / second_num
        else:
            return json.dumps({"error": f"Unsupported operation '{operation}'"})
        return json.dumps({"first_num": first_num, "second_num": second_num,
                           "operation": operation, "result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})




@tool
def get_stock_price(symbol: str) -> str:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage.
    """
    
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=na9xHqmWhFqHth5AckLi"
    r = requests.get(url)
    return r.json()



@tool
def search_pdf(query: str) -> str:
    """
    Search the uploaded PDF document for information relevant to the user's question.
    Use this tool only when the user asks about the content of an uploaded PDF or document.
    """
    return retrieve_context(query)


tools = [search_tool, get_stock_price, calculator, search_pdf]
llm_with_tools = llm.bind_tools(tools) if llm else None

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
SYSTEM_MESSAGE = SystemMessage(content=(
    "You are a helpful AI assistant with access to the following tools:\n"
    "- calculator: for arithmetic operations\n"
    "- get_stock_price: to look up stock prices\n"
    "- duckduckgo_search: to search the web for current information\n"
    "- search_pdf: to retrieve information from an uploaded PDF document\n"
    "Use tools only when needed. Always respond in plain text after receiving tool results."
))

def chat_node(state: ChatState):
    messages = state["messages"]
    # Prepend system message if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SYSTEM_MESSAGE] + list(messages)
    try:
        response = llm_with_tools.invoke(messages)
    except GroqAPIError as e:
        if "Failed to call a function" in str(e):
            # Model generated invalid tool-call JSON — retry with plain LLM
            response = llm.invoke(messages)
        else:
            raise
    return {"messages": [response]}

tool_node = ToolNode(tools)

# -------------------
# 5. Checkpointer
# -------------------


conn = sqlite3.connect("newchatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

