# pyrefly: ignore [missing-import]
import streamlit as st
from langgraph_backend import chatbot, checkpointer, conn
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from rag_backend import process_pdf, get_loaded_filename
import uuid

# =========================== Utilities ===========================

def generate_thread_id():
    return uuid.uuid4()

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get("messages", [])

def delete_thread(thread_id):
    """Remove all checkpoints for this thread from the DB and from session state."""
    cursor = conn.cursor()
    tid = str(thread_id)
    for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
        try:
            cursor.execute(f"DELETE FROM {table} WHERE thread_id = ?", (tid,))
        except Exception:
            pass  # table may not exist depending on langgraph version
    conn.commit()
    # Remove from session list
    if thread_id in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].remove(thread_id)
    # If we deleted the active thread, start a fresh one
 

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()


# ============================ Sidebar ============================
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

# ---------------------- PDF Knowledge Base ----------------------
st.sidebar.divider()
st.sidebar.header("📄 PDF Knowledge Base")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF to enable RAG",
    type="pdf",
    key="pdf_uploader",
)

if uploaded_pdf is not None:
    # Only re-process if a new file is uploaded
    if st.session_state.get("loaded_pdf_name") != uploaded_pdf.name:
        with st.sidebar.status("Processing PDF…", expanded=True) as s:
            st.write(f"📖 Reading `{uploaded_pdf.name}`…")
            n_chunks = process_pdf(uploaded_pdf.read(), uploaded_pdf.name)
            st.session_state["loaded_pdf_name"] = uploaded_pdf.name
            st.session_state["loaded_pdf_chunks"] = n_chunks
            s.update(label="✅ PDF indexed!", state="complete", expanded=False)
    else:
        st.sidebar.success(
            f"✅ **{st.session_state['loaded_pdf_name']}** "
            f"({st.session_state.get('loaded_pdf_chunks', '?')} chunks)"
        )
else:
    if st.session_state.get("loaded_pdf_name"):
        st.sidebar.info(f"📎 Loaded: **{st.session_state['loaded_pdf_name']}**")
    else:
        st.sidebar.caption("No PDF loaded. Upload one to ask questions about it.")

st.sidebar.divider()

st.sidebar.header("My Conversations")
for thread_id in list(st.session_state["chat_threads"])[::-1]:
    col1, col2 = st.sidebar.columns([5, 1])

    with col1:
        is_active = (st.session_state["thread_id"] == thread_id)
        label = f"{'▶ ' if is_active else ''}{str(thread_id)[:8]}…"
        if st.button(label, key=f"thread_{thread_id}", use_container_width=True):
            st.session_state["thread_id"] = thread_id
            messages = load_conversation(thread_id)
            temp_messages = []
            for msg in messages:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                temp_messages.append({"role": role, "content": msg.content})
            st.session_state["message_history"] = temp_messages
            st.rerun()

    # Delete button
    with col2:
        if st.button("X", key=f"delete_{thread_id}", help="Delete this conversation"):
            delete_thread(thread_id)
            st.rerun()

# ============================ Main UI ============================

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
