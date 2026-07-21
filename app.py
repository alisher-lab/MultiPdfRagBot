# app.py — Multi-PDF RAG Chatbot (Qwen2.5-7B via Hugging Face Inference API)
# multi-PDF upload, chunking w/ overlap, embeddings, Chroma storage,
# cross-document retrieval, metadata (doc name/page/category/chunk id),
# document/category filtering, retrieved-chunk debug view, strict
# "not found" fallback, session chat history, and basic logging.

import streamlit as st
import torch
import os
import time
import json
import logging
from datetime import datetime

from dotenv import load_dotenv
from huggingface_hub import HfApi

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

st.set_page_config(page_title="Multi-PDF Chatbot (Qwen2.5-7B)", page_icon="📚", layout="wide")

NOT_FOUND_MSG = "The requested information was not found in the available documents."

# ---------- Logging ----------
# Basic file logging — note: on Streamlit Cloud this file lives on the
# ephemeral container filesystem, so it survives only until the next
# restart/redeploy. Fine for session-level debugging; not durable storage.
logging.basicConfig(
    filename="chat_log.jsonl",
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("pdf_chatbot")


def log_interaction(question, retrieved_meta, response_time, answer, standalone_question=None):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "standalone_question": standalone_question,
        "retrieved_documents": retrieved_meta,
        "response_time_seconds": round(response_time, 2),
        "answer": answer,
    }
    logger.info(json.dumps(entry))
    if "log_entries" not in st.session_state:
        st.session_state.log_entries = []
    st.session_state.log_entries.append(entry)


# ---------- Auth ----------

def validate_hf_token(token: str, retries: int = 2):
    from huggingface_hub.utils import HfHubHTTPError
    api = HfApi()
    last_error = None
    for attempt in range(retries + 1):
        try:
            user_info = api.whoami(token=token)
            return True, user_info.get("name", "unknown user")
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status in (401, 403):
                return False, f"invalid or expired token (HTTP {status})"
            last_error = f"HF server error (HTTP {status}) — likely transient"
        except Exception as e:
            last_error = str(e)
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    return "unverified", last_error


def get_secrets_token():
    try:
        return st.secrets.get("HF_TOKEN")
    except Exception:
        return None


with st.sidebar:
    st.subheader("🔑 Authentication")

    secrets_token = get_secrets_token()
    env_token = os.getenv("HF_TOKEN")

    if secrets_token:
        source, candidate_token = "st.secrets", secrets_token
    elif env_token:
        source, candidate_token = ".env", env_token
    else:
        candidate_token = None

    if candidate_token:
        valid, info = validate_hf_token(candidate_token)
        if valid is True:
            st.success(f"Authenticated via {source} as **{info}**")
            manual_token = candidate_token
        elif valid == "unverified":
            st.warning(f"Couldn't verify via {source} right now ({info}). Proceeding anyway.")
            manual_token = candidate_token
        else:
            st.error(f"Token found in {source} but invalid: {info}")
            st.stop()
    else:
        st.info("No secrets.toml or .env token found — enter one manually.")
        manual_token = st.text_input("Hugging Face Token:", type="password")
        if manual_token:
            valid, info = validate_hf_token(manual_token)
            if valid is True:
                st.success(f"Authenticated as **{info}**")
            elif valid == "unverified":
                st.warning(f"Couldn't verify right now ({info}). Proceeding anyway.")
            else:
                st.error(f"Authentication failed: {info}")
                st.stop()
        else:
            st.warning("Please provide a Hugging Face token to continue.")
            st.stop()

os.environ["HF_TOKEN"] = manual_token


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct:featherless-ai"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Distance threshold for the "not found" guard (the
# no-general-knowledge rule). Chroma's default distance is smaller = more
# similar. Tune this against your own documents — start here and adjust
# based on what you see in the debug "Retrieved Chunks" panel.
NOT_FOUND_DISTANCE_THRESHOLD = 1.3

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using ONLY the information present in the provided context, extracted from the user's uploaded documents. Never use outside knowledge that isn't in the context, even if you happen to know it.

Read the context carefully and synthesize a clear answer, even if the relevant information is spread across a few sentences, phrased differently than the question, or stated as part of a broader category (e.g. a "postgraduate" policy that also covers PhD). You do not need an exact matching sentence — reasoning from clearly related context is fine.

Only respond with exactly this sentence, and nothing else, if the context truly does not address the question's topic at all: "{not_found_msg}"

Context:
{{context}}

Question: {{question}}""".format(not_found_msg=NOT_FOUND_MSG)

PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

CONDENSE_PROMPT_TEMPLATE = """Given the recent conversation and a follow-up question, rewrite the follow-up into a standalone question that includes whatever context it implicitly refers to (e.g. resolve "it", "that", "its" into the actual subject being discussed).

If the follow-up question is already standalone and doesn't depend on the conversation, just return it unchanged.

Respond with ONLY the rewritten question — no explanation, no quotes, no extra text.

Recent conversation:
{chat_history}

Follow-up question: {question}

Standalone question:"""

CONDENSE_PROMPT = PromptTemplate(
    template=CONDENSE_PROMPT_TEMPLATE, input_variables=["chat_history", "question"]
)


def condense_question(llm, chat_history_text: str, question: str) -> str:
    """Rewrites a follow-up question (e.g. 'so what's its requirements?') into
    a standalone one using recent turns, so retrieval isn't done on a
    near-contentless, pronoun-only query. Falls back to the original question
    if the rewrite call fails for any reason — better a slightly worse
    retrieval than a crashed turn."""
    if not chat_history_text.strip():
        return question
    try:
        prompt_text = CONDENSE_PROMPT.format(chat_history=chat_history_text, question=question)
        result = llm.invoke(prompt_text)
        rewritten = result.content.strip().strip('"')
        return rewritten if rewritten else question
    except Exception:
        return question


def format_recent_history(messages, max_turns: int = 3) -> str:
    """Formats the last few chat turns as plain text for the condense prompt.
    Excludes the current in-flight question (already appended by the caller
    before this runs, so we slice it off)."""
    prior = messages[:-1] if messages else []
    recent = prior[-(max_turns * 2):]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# ---------- Cached model resources ----------

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": device})


@st.cache_resource(show_spinner="Connecting to Qwen2.5-7B via Hugging Face Inference API...")
def load_llm(_token: str):
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=_token,
        base_url="https://router.huggingface.co/v1",
        max_tokens=512,
        temperature=0.1,
        streaming=True,
    )


# ---------- Multi-document ingestion ----------

def process_documents(uploaded_files, categories, embeddings):
    """Loads + chunks every uploaded PDF, tags each chunk with metadata
    (document name, page, category, chunk id), and builds one combined
    Chroma collection covering all documents."""
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,  # overlapping technique
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    for uploaded_file in uploaded_files:
        pdf_path = f"/tmp/{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()  # text extraction (requirement 2)
        doc_chunks = splitter.split_documents(pages)  # chunking w/ overlap (requirement 3)

        category = categories.get(uploaded_file.name, "Uncategorized")
        for idx, chunk in enumerate(doc_chunks):
            chunk.metadata["document_name"] = uploaded_file.name
            chunk.metadata["category"] = category
            # page comes from PyPDFLoader already (0-indexed); make it 1-indexed for display
            chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1
            chunk.metadata["chunk_id"] = f"{uploaded_file.name}::p{chunk.metadata['page']}::c{idx}"

        all_chunks.extend(doc_chunks)

    # Embedding generation + storage in vector DB (requirements 4-5).
    # No persist_directory: on Streamlit Cloud the filesystem is ephemeral
    # anyway (see earlier discussion), so an in-memory collection rebuilt
    # each session is simpler and avoids stale leftover DBs.
    vectorstore = Chroma.from_documents(documents=all_chunks, embedding=embeddings)

    doc_names = sorted({c.metadata["document_name"] for c in all_chunks})
    all_categories = sorted({c.metadata["category"] for c in all_chunks})

    return vectorstore, all_chunks, doc_names, all_categories


# ---------- Retrieval across all documents, with optional filtering ----------

def build_filter(selected_docs, selected_categories):
    conditions = []
    if selected_docs:
        conditions.append({"document_name": {"$in": selected_docs}})
    if selected_categories:
        conditions.append({"category": {"$in": selected_categories}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def retrieve_with_scores(vectorstore, question, k, filter_dict):
    results = vectorstore.similarity_search_with_score(question, k=k, filter=filter_dict)
    return results  # list of (Document, distance)


def stream_answer(llm, context, question, retries=2):
    prompt_text = PROMPT.format(context=context, question=question)
    last_error = None
    for attempt in range(retries + 1):
        try:
            got_any_output = False
            for chunk in llm.stream(prompt_text):
                if chunk.content:
                    got_any_output = True
                    yield chunk.content
            return
        except Exception as e:
            last_error = e
            if got_any_output:
                raise
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
    msg = str(last_error)
    if "Gateway Timeout" in msg or "504" in msg or "<!DOCTYPE html>" in msg:
        raise RuntimeError("Hugging Face's inference servers timed out after retries. Try again shortly.")
    raise RuntimeError(msg)


GREETINGS = {"hi", "hii", "hello", "hey", "heya", "yo", "hola",
             "good morning", "good afternoon", "good evening", "greetings"}


def is_greeting(text: str) -> bool:
    return text.strip().lower().strip("!.?") in GREETINGS


def stream_greeting():
    reply = (
        "Hey there! 👋 I'm ready to help you with your documents — "
        "ask me anything about their contents. What would you like to know?"
    )
    for word in reply.split(" "):
        yield word + " "


def stream_static(text):
    for word in text.split(" "):
        yield word + " "


# ---------- App UI ----------

st.title("📚 UMT Admission Guide Chatbot — Qwen2.5-7B")

with st.sidebar:
    st.subheader("📁 Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", type="pdf", accept_multiple_files=True
    )

    categories = {}
    if uploaded_files:
        st.caption("Assign a category to each document:")
        for f in uploaded_files:
            categories[f.name] = st.text_input(
                f"Category — {f.name}", value="General", key=f"cat_{f.name}"
            )

        process_clicked = st.button("Process Documents", type="primary")
        if process_clicked:
            embeddings = load_embeddings()
            with st.spinner("Extracting, chunking, and indexing all documents..."):
                vectorstore, all_chunks, doc_names, all_categories = process_documents(
                    uploaded_files, categories, embeddings
                )
            st.session_state.vectorstore = vectorstore
            st.session_state.doc_names = doc_names
            st.session_state.all_categories = all_categories
            st.session_state.num_chunks = len(all_chunks)
            st.session_state.messages = []  # fresh chat for a fresh document set
            st.success(f"Indexed {len(all_chunks)} chunks across {len(doc_names)} document(s).")

    # Filtering (requirement 10) — only shown once documents are processed
    selected_docs, selected_categories = [], []
    if "vectorstore" in st.session_state:
        st.subheader("🔍 Filter retrieval")
        selected_docs = st.multiselect("By document", st.session_state.doc_names)
        selected_categories = st.multiselect("By category", st.session_state.all_categories)

    st.subheader("⚙️ Settings")
    top_k = st.slider("Chunks to retrieve (k)", min_value=2, max_value=10, value=4)
    show_debug = st.checkbox("Show retrieved-chunk debug panel", value=False)

if "vectorstore" not in st.session_state:
    st.info("Upload PDFs and click **Process Documents** to begin.")
    st.stop()

llm = load_llm(manual_token)
vectorstore = st.session_state.vectorstore

if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat history for the session (requirement 9)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask something about your documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        start_time = time.time()
        retrieved_meta = []
        try:
            if is_greeting(question):
                answer = st.write_stream(stream_greeting())
                elapsed = time.time() - start_time
                log_interaction(question, [], elapsed, answer)
            else:
                filter_dict = build_filter(selected_docs, selected_categories)

                chat_history_text = format_recent_history(st.session_state.messages)
                standalone_question = condense_question(llm, chat_history_text, question)

                with st.spinner("Searching your documents..."):
                    results = retrieve_with_scores(vectorstore, standalone_question, top_k, filter_dict)

                if not results or min(score for _, score in results) > NOT_FOUND_DISTANCE_THRESHOLD:
                    # Hard guard (requirement 12 + no-general-knowledge rule):
                    # nothing relevant enough was retrieved — don't even call
                    # the LLM, just return the required fixed message.
                    answer = NOT_FOUND_MSG
                    st.write(answer)
                else:
                    context = "\n\n".join(doc.page_content for doc, _ in results)
                    answer = st.write_stream(stream_answer(llm, context, standalone_question))

                # Document name + page number shown with every answer (requirement 8)
                with st.expander("📄 Sources"):
                    for doc, score in results:
                        st.markdown(
                            f"**{doc.metadata.get('document_name', '?')}** — "
                            f"page {doc.metadata.get('page', '?')} "
                            f"(category: {doc.metadata.get('category', '?')}, "
                            f"distance: {score:.3f})"
                        )

                # Retrieved-chunk debug display (requirement 11)
                if show_debug:
                    with st.expander("🛠️ Debug: Retrieved Chunks"):
                        if standalone_question != question:
                            st.caption(f"Rewritten for retrieval: *{standalone_question}*")
                        for doc, score in results:
                            st.markdown(f"**chunk_id:** `{doc.metadata.get('chunk_id')}`  |  **distance:** {score:.3f}")
                            st.text(doc.page_content[:500])
                            st.markdown("---")

                retrieved_meta = [
                    {
                        "document_name": doc.metadata.get("document_name"),
                        "page": doc.metadata.get("page"),
                        "category": doc.metadata.get("category"),
                        "chunk_id": doc.metadata.get("chunk_id"),
                        "distance": round(score, 4),
                    }
                    for doc, score in results
                ]
                elapsed = time.time() - start_time
                log_interaction(question, retrieved_meta, elapsed, answer, standalone_question)

        except Exception as e:
            answer = f"Error calling the model: {e}"
            st.error(answer)
            elapsed = time.time() - start_time
            log_interaction(question, retrieved_meta, elapsed, answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Session log viewer (supports requirement 13's visibility, optional but handy)
with st.sidebar:
    if st.session_state.get("log_entries"):
        with st.expander("📊 Session Logs"):
            for entry in reversed(st.session_state.log_entries[-10:]):
                st.caption(f"{entry['timestamp']} — {entry['response_time_seconds']}s")
                st.text(entry["question"])
