# langgraph_backend.py

import os
import json
import sqlite3
import tempfile
from typing import Annotated, TypedDict

import requests
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI  # pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq  # pyrefly: ignore [missing-import]
from groq import APIError as GroqAPIError  # pyrefly: ignore [missing-import]
from langchain_community.tools import DuckDuckGoSearchRun  # pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader  # pyrefly: ignore [missing-import]
from langchain_community.vectorstores import FAISS  # pyrefly: ignore [missing-import]
from langchain_community.embeddings import HuggingFaceEmbeddings  # pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter  # pyrefly: ignore [missing-import]
from langgraph.checkpoint.sqlite import SqliteSaver  # pyrefly: ignore [missing-import]
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import asyncio

load_dotenv()

# ---------------------------------------------------------------------------
# 1. LLM
# ---------------------------------------------------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_retries=2,
)

# ---------------------------------------------------------------------------
# 2. RAG — FAISS vector store (global, single document at a time)
# ---------------------------------------------------------------------------
_faiss_retriever = None  # set by ingest_pdf()
_loaded_filename: str = ""

_FAISS_DIR = "faiss_store"
_FAISS_META = "faiss_store/filename.txt"

# Auto-load persisted index on startup so hot-reloads don't wipe state
_HF_EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
try:
    if os.path.exists(_FAISS_DIR):
        print("[RAG] Found faiss_store/ — loading index from disk...")
        _vs = FAISS.load_local(_FAISS_DIR, _HF_EMBEDDINGS, allow_dangerous_deserialization=True)
        _faiss_retriever = _vs.as_retriever(search_kwargs={"k": 4})
        if os.path.exists(_FAISS_META):
            with open(_FAISS_META, encoding="utf-8") as _f:
                _loaded_filename = _f.read().strip()
        print(f"[RAG] ✅ Loaded index for: '{_loaded_filename}'")
    else:
        print("[RAG] No faiss_store/ found — waiting for PDF upload.")
except Exception as e:
    print(f"[RAG] ⚠️ Failed to load index from disk: {e}")


def ingest_pdf(file_bytes: bytes, filename: str = "") -> int:
    """
    Load a PDF from raw bytes, split it, embed with OpenAI, store in FAISS.
    Saves the index to disk so it survives hot-reloads.
    Returns the number of chunks created.
    """
    global _faiss_retriever, _loaded_filename

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(docs)

        embeddings = _HF_EMBEDDINGS
        vector_store = FAISS.from_documents(chunks, embeddings)

        # Persist to disk
        os.makedirs(_FAISS_DIR, exist_ok=True)
        vector_store.save_local(_FAISS_DIR)
        with open(_FAISS_META, "w", encoding="utf-8") as f:
            f.write(filename or os.path.basename(tmp_path))

        _faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        _loaded_filename = filename or os.path.basename(tmp_path)

        return len(chunks)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def get_loaded_filename() -> str:
    return _loaded_filename


# ---------------------------------------------------------------------------
# 3. Tools
# ---------------------------------------------------------------------------
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
        return json.dumps({"result": result, "operation": operation,
                           "first_num": first_num, "second_num": second_num})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_stock_price(symbol: str) -> str:
    """
    Fetch the latest stock price for a ticker symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=na9xHqmWhFqHth5AckLi"
    )
    try:
        return json.dumps(requests.get(url, timeout=10).json())
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def search_pdf(query: str) -> str:
    """
    Search the uploaded PDF document for information relevant to the user's question.
    Use this tool only when the user asks about the content of the uploaded PDF or document.
    """
    if _faiss_retriever is None:
        return "No PDF has been uploaded yet. Please upload a PDF first."
    docs = _faiss_retriever.invoke(query)
    if not docs:
        return "No relevant content found in the document."
    return "\n\n".join(doc.page_content for doc in docs)


tools = [search_tool, get_stock_price, calculator, search_pdf]
llm_with_tools = llm.bind_tools(tools)

# ---------------------------------------------------------------------------
# 4. State
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# 5. Graph nodes
# ---------------------------------------------------------------------------
def _build_system_message() -> SystemMessage:
    filename = _loaded_filename
    if filename:
        pdf_context = (
            f"A PDF document has been uploaded by the user: '{filename}'. "
            "When the user asks anything about the document, its contents, or any data in it, "
            "you MUST call the `search_pdf` tool to retrieve the relevant information. "
            "Do NOT say you don't have a document — one is available."
        )
    else:
        pdf_context = (
            "No PDF has been uploaded yet. "
            "If the user asks about a document, politely ask them to upload a PDF using the sidebar."
        )

    return SystemMessage(content=(
        "You are a helpful AI assistant. You have access to these tools:\n"
        "- calculator: arithmetic operations\n"
        "- get_stock_price: look up stock prices\n"
        "- duckduckgo_search: search the web\n"
        "- search_pdf: search an uploaded PDF document\n\n"
        f"{pdf_context}\n\n"
        "Use tools only when needed. Respond in plain, concise text."
    ))


def chat_node(state: ChatState):
    messages = state["messages"]
    system_msg = _build_system_message()
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [system_msg] + list(messages)
    else:
        messages = [system_msg] + list(messages[1:])
    try:
        response = llm_with_tools.invoke(messages)
    except GroqAPIError as e:
        if "Failed to call a function" in str(e):
            response = llm.invoke(messages)
        else:
            raise
    return {"messages": [response]}


tool_node = ToolNode(tools)

# ---------------------------------------------------------------------------
# 6. Checkpointer + Graph
# ---------------------------------------------------------------------------
conn = sqlite3.connect("newchatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# ---------------------------------------------------------------------------
# 7. Thread helpers
# ---------------------------------------------------------------------------
def retrieve_all_threads() -> list:
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)
