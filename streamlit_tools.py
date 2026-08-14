# streamlit_tools.py — Main Streamlit UI (updated for Gemini + Blog Agent)

import os
import uuid
from datetime import date
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()

# ---------------------------------------------------------------------------
# Lazy-load the compiled app (avoids heavy init before the page renders)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI engine…")
def _get_app():
    from graph.build_graph import build_app
    return build_app()


@st.cache_resource(show_spinner=False)
def _get_conn():
    from graph.memory import get_connection
    return get_connection()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def generate_thread_id():
    return str(uuid.uuid4())


def retrieve_all_threads():
    try:
        from graph.memory import get_checkpointer
        cp = get_checkpointer()
        return list({
            c.config["configurable"]["thread_id"]
            for c in cp.list(None)
        })
    except Exception:
        return []


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)
    st.session_state["message_history"] = []


def load_conversation(thread_id):
    try:
        app = _get_app()
        state = app.get_state(config={"configurable": {"thread_id": thread_id}})
        return state.values.get("messages", [])
    except Exception:
        return []


def delete_thread(thread_id):
    conn = _get_conn()
    if conn is None:
        return
    cursor = conn.cursor()
    tid = str(thread_id)
    for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
        try:
            cursor.execute(f"DELETE FROM {table} WHERE thread_id = ?", (tid,))
        except Exception:
            pass
    conn.commit()
    if thread_id in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].remove(thread_id)


# ---------------------------------------------------------------------------
# Session initialisation
# ---------------------------------------------------------------------------
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🤖 AI Chatbot + Blog Agent")
st.sidebar.caption("Powered by Google Gemini · Tavily · LangGraph")

if st.sidebar.button("✨ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# ---------- PDF Knowledge Base ----------
st.sidebar.divider()
st.sidebar.header("📄 PDF Knowledge Base")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF to enable RAG",
    type="pdf",
    key="pdf_uploader",
)

if uploaded_pdf is not None:
    if st.session_state.get("loaded_pdf_name") != uploaded_pdf.name:
        with st.sidebar.status("Processing PDF…", expanded=True) as s:
            st.write(f"📖 Reading `{uploaded_pdf.name}`…")
            import rag_backend
            n_chunks = rag_backend.process_pdf(uploaded_pdf.read(), uploaded_pdf.name)
            st.session_state["loaded_pdf_name"] = uploaded_pdf.name
            st.session_state["loaded_pdf_chunks"] = n_chunks
            s.update(label=f"✅ PDF indexed! ({n_chunks} chunks)", state="complete", expanded=False)
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

# ---------- Conversation list ----------
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
                if isinstance(msg, HumanMessage):
                    temp_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage) and msg.content:
                    temp_messages.append({"role": "assistant", "content": msg.content})
            st.session_state["message_history"] = temp_messages
            st.rerun()
    with col2:
        if st.button("✕", key=f"delete_{thread_id}", help="Delete conversation"):
            delete_thread(thread_id)
            st.rerun()

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("💬 AI Assistant")
st.caption("Ask me anything, upload a PDF to chat with it, or say **'write a blog about ...'** to generate a full blog post!")

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_input = st.chat_input("Type here · e.g. 'write a blog about quantum computing'")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Detect likely blog request for UI hint (rough check, real classification happens in graph)
    is_likely_blog = any(
        kw in user_input.lower()
        for kw in ("write a blog", "write blog", "create a blog", "draft a blog",
                   "blog about", "blog on", "article about", "create an article")
    )

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "tags": ["blog"] if is_likely_blog else ["chat"],
        "metadata": {
            "pdf_uploaded": bool(st.session_state.get("loaded_pdf_name")),
            "thread_id": st.session_state["thread_id"],
        },
        "run_name": "blog_turn" if is_likely_blog else "chat_turn",
    }

    app = _get_app()

    with st.chat_message("assistant"):
        if is_likely_blog:
            # Blog generation: invoke (not stream) with multi-step status updates
            with st.status("🔍 Researching topic…", expanded=True) as status:
                st.write("📊 Routing & research in progress…")
                try:
                    result = app.invoke(
                        {
                            "messages": [HumanMessage(content=user_input)],
                            "thread_id": st.session_state["thread_id"],
                            "as_of": str(date.today()),
                        },
                        config=CONFIG,
                    )
                    status.update(label="✅ Blog generated!", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="❌ Error", state="error", expanded=True)
                    st.error(f"Error generating blog: {e}")
                    result = None

            if result:
                final_md = result.get("final") or ""
                last_msg = result.get("messages", [{}])[-1]
                display_content = final_md or (last_msg.content if hasattr(last_msg, "content") else "")

                st.markdown(display_content)

                # Show generated images if any
                images_dir = Path("data/images")
                if images_dir.exists():
                    new_images = sorted(images_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)[:3]
                    if new_images:
                        st.divider()
                        st.caption("📸 Generated Images")
                        for img_path in new_images:
                            st.image(str(img_path), use_container_width=True)

                # Download button
                if display_content:
                    st.download_button(
                        label="⬇️ Download blog.md",
                        data=display_content,
                        file_name="blog.md",
                        mime="text/markdown",
                    )

                ai_message = display_content
            else:
                ai_message = "[Blog generation failed — please try again]"

        else:
            # Normal chat: stream token-by-token
            status_holder = {"box": None}

            def ai_only_stream():
                for chunk in app.stream(
                    {
                        "messages": [HumanMessage(content=user_input)],
                        "thread_id": st.session_state["thread_id"],
                        "as_of": str(date.today()),
                    },
                    config=CONFIG,
                    stream_mode="messages",
                ):
                    # LangGraph v1: stream yields raw message objects or (msg, meta) tuples
                    message_chunk = chunk[0] if isinstance(chunk, tuple) else chunk

                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(
                                f"Using `{tool_name}`...", expanded=True
                            )
                        else:
                            status_holder["box"].update(
                                label=f"Using `{tool_name}`...",
                                state="running",
                                expanded=True,
                            )
                    if isinstance(message_chunk, AIMessage) and message_chunk.content:
                        yield message_chunk.content

            try:
                ai_message = st.write_stream(ai_only_stream())
            except Exception as e:
                err = str(e)
                if "429" in err or "rate_limit" in err or "quota" in err:
                    st.warning(
                        "⏳ **API rate limit reached.** Please wait a moment and try again.\n\n"
                        "Tip: Check your Google AI Studio quota or try again in a few seconds."
                    )
                    ai_message = "[Rate limit — please retry]"
                else:
                    st.error(f"❌ Error: {err}")
                    ai_message = f"[Error: {err}]"

            if status_holder["box"] is not None:
                status_holder["box"].update(label="✅ Done", state="complete", expanded=False)

    # Save assistant message to history
    if ai_message:
        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )
