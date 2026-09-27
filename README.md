## 3. Detailed Component Architecture

### A. Presentation Layer — `streamlit_tools.py`

The presentation layer provides the Streamlit-based user interface and manages the lifecycle of conversations.

**Session Lifecycle**

* Automatically generates a unique `thread_id` using UUID.
* Stores the `thread_id` in `st.session_state` to maintain conversation identity.

**State Caching**

* Uses `@st.cache_resource` to cache the LangGraph application instance and database connections.
* Prevents unnecessary re-initialization and improves application startup performance.

**Dual Execution Modes**

**1. Streaming Chat Mode**

* Uses `app.stream(..., stream_mode="messages")`.
* Displays LLM responses token-by-token using `st.write_stream()`.
* Provides real-time visibility into tool execution and intermediate states.

**2. Status-Tracked Blog Mode**

* Uses `app.invoke()` for the complete blog-generation workflow.
* Uses Streamlit status components to display progress through:

  * Routing
  * Research
  * Drafting
  * Image generation

---

### B. Intent Router — `graph/intent_router.py`

The intent router determines which workflow should handle the user's request.

* Uses the lightweight `openai/gpt-oss-20b` model.
* Uses a low temperature (`0.0`) for consistent classification.
* Inspects the latest user message and classifies it into two categories:

| Intent | Description                                                            |
| ------ | ---------------------------------------------------------------------- |
| `blog` | Requests to write articles, guides, essays, or blog posts              |
| `chat` | Questions, conversations, calculations, PDF queries, and general tasks |

The router then directs the request to the appropriate LangGraph subgraph.

---

### C. Conversational Agent Subgraph — `graph/chat/`

The chat workflow follows a standard **ReAct (Reasoning + Acting) cycle**:

```text
User Input
    ↓
Agent
    ↓
Tool Selection
    ↓
Tool Execution
    ↓
Agent
    ↓
END
```

#### Tool Calling

The conversational agent can use external tools when required:

* **`tavily_search`** — Searches the live web for current or real-time information.
* **`search_pdf`** — Retrieves relevant context from uploaded PDFs using `rag_backend.retrieve_context(query, k=4)`.

This allows the conversational agent to combine general LLM reasoning with external information and document-specific retrieval.

---

### D. Multi-Agent Blog Generation Subgraph — `graph/blog/`

The blog workflow is implemented as a multi-agent pipeline consisting of routing, research, planning, parallel section generation, reduction, and image generation.

#### 1. Router Node — `router_node.py`

Determines whether the requested topic requires external research or can be handled as a general closed-book topic.

```text
User Topic
    ↓
Router
    ├── Closed-book
    └── Requires Research
```

#### 2. Research Node — `research_node.py`

When research is required, this node:

1. Generates search queries.
2. Searches Tavily and/or uploaded PDFs.
3. Extracts and parses relevant passages.
4. Builds an evidence dossier for downstream agents.

The evidence dossier provides curated information that the writing agents can use while generating the blog.

#### 3. Orchestrator Node — `orchestrator_node.py`

Creates a structured `BlogPlan` containing the sections required for the article.

It then uses LangGraph's `Send()` mechanism to dispatch individual section-writing tasks to worker nodes in parallel.

```text
                 ┌── Worker → Section 1
                 │
Orchestrator ────┼── Worker → Section 2
                 │
                 └── Worker → Section 3
```

This enables parallel generation of independent blog sections.

#### 4. Worker Node — `worker_node.py`

Each worker is responsible for writing one specific section.

The worker receives:

* Section requirements
* Writing tone and constraints
* Relevant evidence snippets
* The overall blog context

It then produces the requested section while following the provided constraints.

#### 5. Reducer Pipeline — `reducer.py`

The reducer combines the outputs from the parallel workers and performs the final content-processing steps.

**`merge_content`**

* Combines generated sections.
* Restores the original section order.

**`decide_images`**

* Uses structured LLM output (`GlobalImagePlan`) to determine where visual content should be placed.
* Creates 1–3 image placeholders such as:

```text
{{IMAGE_1}}
{{IMAGE_2}}
```

**`generate_and_place_images`**

* Generates images using Pollinations.ai.
* Saves generated `.png` files under `data/images/`.
* Replaces image placeholders with Markdown image syntax.

Overall workflow:

```text
Router
   ↓
Research
   ↓
Orchestrator
   ↓
Parallel Workers
   ↓
Merge Content
   ↓
Image Planning
   ↓
Image Generation
   ↓
Final Blog
```

---

### E. In-Memory RAG Engine — `rag_backend.py`

The RAG engine provides document ingestion, chunking, indexing, and retrieval for uploaded PDFs.

#### Document Parsing

Uses `pypdf.PdfReader` to extract text from PDFs on a page-by-page basis.

```text
PDF
 ↓
PdfReader
 ↓
Page-level text
```

#### Chunking

The extracted text is divided using a **sliding-window chunking strategy**.

Configuration:

| Parameter  |     Value |
| ---------- | --------: |
| Chunk size | 150 words |
| Overlap    |  20 words |
| Step size  | 130 words |

For example:

```text
Chunk 1: words 1–150
Chunk 2: words 131–
```
<img width="4047" height="8192" alt="Text Chunking Process Model-2026-09-27-044450" src="https://github.com/user-attachments/assets/cc39add4-0db0-4a56-b947-de4fbce819d4" />
