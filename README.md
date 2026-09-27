![Uploading image.png…]()

3. Detailed Component Architecture
A. Presentation Layer (

streamlit_tools.py
)
Session Lifecycle: Automatically creates unique thread_id UUIDs stored in st.session_state.
State Caching: Uses @st.cache_resource for the LangGraph instance (_get_app()) and database connections (_get_conn()) to prevent cold-start reloads.
Dual Execution Modes:
Streaming Chat Mode: Uses app.stream(..., stream_mode="messages") combined with st.write_stream() for token-by-token visual feedback and real-time tool execution status boxes.
Status-Tracked Blog Mode: Uses app.invoke(...) with st.status() to track routing, research, drafting, and image generation phases.
B. Intent Router (

graph/intent_router.py
)
Uses the lightweight openai/gpt-oss-20b model with low temperature (0.0).
Inspects the latest user input and classifies it into:
"blog": For explicit requests to write articles, guides, essays, or blog posts.
"chat": For all questions, chit-chat, calculations, PDF queries, and general tasks.
C. Conversational Agent Subgraph (

graph/chat/
)
Architecture: Standard ReAct cycle (agent $\rightarrow$ tools $\rightarrow$ agent $\rightarrow$ END).
Tool Calling:
tavily_search: Searches the live web when questions involve real-time info or current events.
search_pdf: Calls rag_backend.retrieve_context(query, k=4) to extract context from any uploaded document.
D. Multi-Agent Blog Generation Subgraph (

graph/blog/
)
1. Router Node (

router_node.py
): Determines if external research is needed or if it is a general closed-book topic.
2. Research Node (

research_node.py
): Generates search queries, searches Tavily/PDF, parses passages, and builds an evidence dossier.
3. Orchestrator Node (

orchestrator_node.py
): Generates a structured outline (BlogPlan) and uses LangGraph Send("worker", ...) to dispatch section writing jobs in parallel.
4. Worker Node (

worker_node.py
): Writes a specific section adhering to tone, constraints, and curated evidence snippets.
5. Reducer Pipeline (

reducer.py
):
merge_content: Stitches section results in index order.
decide_images: Uses structured LLM output (GlobalImagePlan) to place 1–3 visual anchor placeholders ({{IMAGE_1}}).
generate_and_place_images: Calls Pollinations.ai, saves .png files to data/images/, and replaces placeholders with markdown image syntax.
E. In-Memory RAG Engine (

rag_backend.py
)
Document Parsing: pypdf.PdfReader extracts page-by-page text.
Chunking: Sliding window chunker configured with:
Chunk size: 150 words
Overlap: 20 words
Step size: 130 words
Indexing & Retrieval:
Custom term-frequency (TF) and smoothed inverse document frequency (IDF) matrix.
L2-normalized vectors; retrieval via matrix-vector dot product (cosine similarity) returning Top-$k$ passages with page numbers.
F. Memory & Persistence (

graph/memory.py
)
Uses SqliteSaver connected to newchatbot.db.
Stores conversation histories across sessions keyed by thread_id so conversations can be reloaded and resumed.
