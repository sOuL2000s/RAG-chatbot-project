import sys
import os
import uuid
from typing import List, Dict, Tuple
from dataclasses import dataclass

import streamlit as st
import pandas as pd

# PDF
from pypdf import PdfReader

# Vector DB
import chromadb

# Google Generative AI SDK
import google.generativeai as genai

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# ---------- Utils ------------
# -----------------------------

def get_gemini_client():
    """Configures Google Gemini client using API key from .env or secrets."""
    load_dotenv(override=True)
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        key = st.secrets.get("GEMINI_API_KEY")
    
    if not key:
        st.error("No Gemini API key found. Please set GEMINI_API_KEY in your .env file or Streamlit secrets.")
        st.stop()
    
    genai.configure(api_key=key.strip())
    return genai

def new_uuid() -> str:
    return str(uuid.uuid4())

def safe_clean(s: str) -> str:
    """Removes problematic characters and excessive whitespace."""
    if not s:
        return ""
    cleaned = s.replace("\x00", " ").replace("\r", " ").replace("\n", " ")
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()

# --- MODIFIED CHUNKING FUNCTION ---
def chunk_text(
    text: str,
    target_chunk_chars: int = 3000,
    max_chunk_chars: int = 3500, # A hard limit to prevent excessively large chunks
    overlap_chars: int = 200
) -> List[str]:
    """
    Splits text into character-based chunks with overlap, prioritizing natural breaks.
    
    Tries to split by paragraphs first, then falls back to character-based splitting
    if paragraphs are too large or the text isn't paragraph-structured.
    """
    if not text or not text.strip():
        return []

    text = safe_clean(text) # Ensure text is clean before chunking

    candidate_chunks = []
    
    # Attempt 1: Split by paragraphs (double newline)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    current_chunk = ""
    for para in paragraphs:
        # If adding the current paragraph exceeds the max_chunk_chars,
        # finalize the current_chunk (if not empty) and start a new one.
        if len(current_chunk) + len(para) + 2 > max_chunk_chars: # +2 for potential newlines if joined later
            if current_chunk.strip():
                candidate_chunks.append(current_chunk.strip())
            current_chunk = para # Start new chunk with this paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
    
    if current_chunk.strip():
        candidate_chunks.append(current_chunk.strip())

    final_chunks = []
    # Now, process candidate_chunks, ensuring none exceed max_chunk_chars and adding overlap
    for i, candidate in enumerate(candidate_chunks):
        if len(candidate) > max_chunk_chars:
            # If a candidate chunk (e.g., a single very long paragraph) is still too large,
            # fall back to character-based splitting for this specific candidate.
            sub_chunks = []
            start_idx = 0
            while start_idx < len(candidate):
                end_idx = min(start_idx + target_chunk_chars, len(candidate))
                sub_chunk = candidate[start_idx:end_idx]
                if sub_chunk.strip():
                    sub_chunks.append(sub_chunk.strip())
                if end_idx == len(candidate):
                    break
                start_idx += (target_chunk_chars - overlap_chars)
                if start_idx < 0:
                    start_idx = 0 # Safety check
            final_chunks.extend(sub_chunks)
        else:
            # If the candidate chunk is a reasonable size, just add it.
            final_chunks.append(candidate)
            
            # Add overlap logic for paragraphs if they are relatively small
            # This is an alternative to standard overlap: re-add part of previous paragraph to next.
            # Simpler approach: let the vector DB handle proximity for now with distinct chunks.
            # For strict overlap on paragraph-based, this logic gets complex fast.
            # For now, let's keep chunks distinct but ensure they are not too large.

    print(f"Debug: Initial text length: {len(text)} chars")
    print(f"Debug: Produced {len(final_chunks)} chunks.")
    # for i, ch in enumerate(final_chunks):
    #     print(f"  Chunk {i+1} ({len(ch)} chars): {ch[:100]}...") # Print first 100 chars of each chunk
    
    return final_chunks
# --- END MODIFIED CHUNKING FUNCTION ---

def read_pdf(file) -> list[tuple[str, dict]]:
    """Extract text from a PDF. OCR fallback is disabled by default."""
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((text, {"source": file.name, "type": "pdf", "page": i+1}))
        else:
            pages.append(("", {"source": file.name, "type": "pdf", "page": i+1}))
    return pages

def read_csv(file) -> List[Tuple[str, Dict]]:
    """Returns (row_text, metadata) per row for CSV files."""
    try:
        df = pd.read_csv(file)
        rows = []
        for idx, row in df.iterrows():
            row_values = []
            for col in df.columns:
                val = row[col]
                if pd.notna(val):
                    row_values.append(f"{col}: {val}")
            
            if row_values:
                row_text = " | ".join(row_values)
                rows.append((row_text, {"source": file.name, "type": "csv", "row": int(idx) + 1}))
        return rows
    except Exception as e:
        st.error(f"Error reading CSV {file.name}: {e}")
        return []

def read_xlsx(file) -> List[Tuple[str, Dict]]:
    """Returns (row_text, metadata) per row for XLSX/XLS files."""
    try:
        df = pd.read_excel(file)
        rows = []
        for idx, row in df.iterrows():
            row_values = []
            for col in df.columns:
                val = row[col]
                if pd.notna(val):
                    row_values.append(f"{col}: {val}")
            
            if row_values:
                row_text = " | ".join(row_values)
                rows.append((row_text, {"source": file.name, "type": "xlsx", "row": int(idx) + 1}))
        return rows
    except Exception as e:
        st.error(f"Error reading Excel {file.name}: {e}")
        return []

def read_text(file) -> List[Tuple[str, Dict]]:
    """Reads content from a text-based file."""
    try:
        content = file.getvalue().decode("utf-8")
        return [(content, {"source": file.name, "type": "txt", "page": 1})]
    except Exception as e:
        st.error(f"Error reading text file {file.name}: {e}")
        return []

@dataclass
class RAGChunk:
    id: str
    text: str
    metadata: Dict

# -----------------------------
# ------ Vector Store ---------
# -----------------------------

def get_chroma_client():
    """Creates an in-memory ChromaDB client for temporary storage."""
    try:
        client = chromadb.Client()
        return client
    except Exception as e:
        st.error(f"Could not create in-memory ChromaDB client: {e}")
        return None

def get_or_create_collection(chroma_client, collection_name: str = "kb_scout_documents"):
    """Get existing collection or create new one."""
    try:
        return chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        st.error(f"Error getting/creating collection: {e}")
        return None

def embed_texts(texts: List[str], model: str = "models/embedding-001", batch_size: int = 100) -> List[List[float]]:
    """Batches embeddings using Gemini's embedding model."""
    if not texts:
        return []
    
    all_embeddings: List[List[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_num = (i // batch_size) + 1
        # st.write(f"Processing embedding batch {batch_num}/{total_batches}...") # Uncomment for debug
        
        batch = texts[i:i + batch_size]
        try:
            response = genai.embed_content(model=model, content=batch)
            batch_embeddings = [item['values'] for item in response['embedding']]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            st.error(f"Error creating embeddings for batch {batch_num}: {e}")
            return []
    
    return all_embeddings

def add_chunks_to_collection(collection, rag_chunks: List[RAGChunk]):
    """Add chunks to in-memory collection."""
    if not rag_chunks or not collection:
        return False
    
    valid_chunks = [c for c in rag_chunks if c.text and c.text.strip()]
    if not valid_chunks:
        st.info("No valid text chunks to add after cleaning.") # Added for clarity
        return False
    
    documents = [c.text for c in valid_chunks]
    metadatas = [c.metadata for c in valid_chunks]
    ids = [c.id for c in valid_chunks]

    st.write(f"Attempting to embed {len(documents)} documents for ChromaDB...")
    embeddings = embed_texts(documents)
    
    if not embeddings:
        st.error("Embedding process failed to return any embeddings. Please check API key and quota.")
        return False
    if len(embeddings) != len(documents):
        st.error(f"Mismatch in embedding length: Expected {len(documents)}, got {len(embeddings)}. This might indicate partial embedding failure or an API issue.")
        return False

    try:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        return True
    except Exception as e:
        st.error(f"Error adding to collection: {e}")
        return False

def retrieve(collection, query: str, top_k: int = 6) -> List[Tuple[str, Dict, float]]:
    if not collection:
        return []
    
    count = collection.count()
    if count == 0:
        return []
    
    try:
        q_emb = embed_texts([query])
        if not q_emb:
            return []
        
        res = collection.query(
            query_embeddings=q_emb,
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"]
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        
        scored = list(zip(docs, metas, dists))
        scored.sort(key=lambda x: x[2])
        return scored
    except Exception as e:
        st.error(f"Error during retrieval: {e}")
        return []

def format_context(snippets: List[Tuple[str, Dict, float]]) -> str:
    """Formats retrieved snippets into a string for LLM context."""
    blocks = []
    for i, (doc, meta, dist) in enumerate(snippets, 1):
        src = meta.get("source", "unknown")
        loc_parts = []
        if meta.get("type") == "pdf":
            loc_parts.append(f"page {meta.get('page', 'unknown')}")
        elif "row" in meta:
            loc_parts.append(f"row {meta.get('row', 'unknown')}")
        else:
            loc_parts.append(meta.get('type', 'document'))

        location_str = ", ".join(loc_parts)
        blocks.append(f"[{i}] Source: {src} ({location_str})\n{doc}")
    return "\n\n".join(blocks)

def get_uploaded_files_from_collection(collection):
    """Get list of unique files that have been uploaded to the collection."""
    if not collection:
        return []
    
    try:
        all_data = collection.get(include=["metadatas"])
        metadatas = all_data.get("metadatas", [])
        
        files = set()
        for meta in metadatas:
            if "source" in meta and "type" in meta:
                files.add((meta["source"], meta["type"]))
        
        return list(files)
    except:
        return []

SYSTEM_PROMPT = """You are a helpful AI assistant. Your goal is to provide comprehensive and accurate answers.
You have access to a vast general knowledge base. Additionally, I will provide you with specific context relevant to the user's question, enclosed in <context> tags.

Here's how you should approach answering:
1.  **Prioritize the provided context:** If the user's question can be directly and accurately answered by the information within the <context> tags, use that information. When information from the context is used, cite your sources clearly using the bracket numbers provided (e.g., [1], [2]).
2.  **Augment with your general knowledge:** If the provided context is insufficient, irrelevant, or does not directly address parts of the question, supplement your answer with your own general knowledge.
3.  **Combine information:** For questions that require both specific details from the context and general knowledge (e.g., "What is the capital of France and what is Project Alpha?"), seamlessly integrate information from both sources.
4.  **State limitations:** If you cannot find relevant information in the provided context AND your general knowledge is also insufficient for a specific part of the question, politely state that you do not have enough information to answer that specific part.
5.  **Be helpful, friendly, and professional.**
"""

def answer_with_rag(question: str, context_text: str):
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    full_prompt = f"{SYSTEM_PROMPT}\n\n<context>\n{context_text}\n</context>\n\nUser Question: {question}\nAnswer:"

    gemini_chat_history = []
    for msg in st.session_state.history[:-1]:
        gemini_chat_history.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [{"text": msg["content"]}]
        })
    
    chat_session = model.start_chat(history=gemini_chat_history)
    
    response_stream = chat_session.send_message(full_prompt, stream=True)
    
    return response_stream


# -----------------------------
# --------- UI Layer ----------
# -----------------------------

st.set_page_config(
    page_title="K&B Scout AI Enterprise Assistant", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for adaptive light/dark theme styling and visual appeal
st.markdown(
    """
    <style>
        /* Base styles for the Streamlit app. Ensure the overall app container
           is a flex column to properly distribute vertical space. */
        div[data-testid="stAppViewContainer"] {
            display: flex;
            flex-direction: column;
            min-height: 100vh; /* Make the app take full viewport height */
            padding: 0 !important;
            margin: 0 !important;
        }

        /* Define custom CSS variables for consistent theming */
        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --header-text-color: white;
            --border-radius: 12px;
            --shadow-subtle: 0 1px 3px rgba(0,0,0,0.05);
            --shadow-medium: 0 4px 12px rgba(102, 126, 234, 0.4);
            --shadow-strong: 0 6px 16px rgba(102, 126, 234, 0.5);
            --st-primary-color: #667eea; /* Align Streamlit primary color with our design */
            --st-secondary-text: #6c757d; /* Consistent secondary text color */
        }

        /* Light Mode specific colors (defaults for [data-theme="light"]) */
        [data-theme="light"] {
            --background-color: #ffffff; /* Overall app background */
            --content-bg: #ffffff; /* Background for main content areas */
            --content-text-color: #333333; /* Dark text for content area */
            --border-color: #e9ecef;
            --chat-bg-assistant: #f8f9fa; /* Light grey for assistant bubble */
            --chat-bg-user: #e6f7ff; /* Light blue for user bubble */
            --file-item-bg: #ffffff;
            --component-bg-color: #f0f2f6; /* Streamlit's default light gray for some elements */
        }

        /* Dark Mode specific colors */
        [data-theme="dark"] {
            --background-color: #0e1117; /* Streamlit's default dark background */
            --content-bg: #1e1e1e; /* Dark background for main content areas */
            --content-text-color: #e0e0e0; /* Light text color for dark mode */
            --border-color: #333333; /* Darker border */
            --chat-bg-assistant: #2d2d2d; /* Slightly lighter dark for assistant bubble */
            --chat-bg-user: #3c3c3c; /* A bit lighter dark for user bubble */
            --file-item-bg: #282828;
            --component-bg-color: #262730; /* Streamlit's default dark gray for some elements */
        }

        /* --- Global Streamlit Overrides --- */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;} /* Hide default Streamlit header bar */
        
        /* Remove default Streamlit block container padding globally */
        .block-container {
            padding: 1rem 1rem !important; /* Restore a small default padding, adjust as needed */
        }
        
        /* --- General App Structure & Theming --- */
        /* Apply background and text color to the entire app content area */
        div[data-testid="stAppViewContainer"] > .main {
            background-color: var(--background-color) !important;
            color: var(--content-text-color) !important;
            padding: 0; /* Important to remove Streamlit's default main padding for custom layout */
            margin: 0;
            flex-grow: 1; /* Allow main content to grow */
            overflow: hidden; /* Prevent main .block-container from scrolling independently */
            display: flex; /* Make it a flex container for its children (columns and chat input) */
            flex-direction: column; /* Stack children vertically */
        }
        
        /* Our custom header */
        .app-header {
            background: var(--primary-gradient);
            color: var(--header-text-color);
            padding: 20px 30px;
            display: flex;
            align-items: center;
            gap: 15px;
            flex-shrink: 0; /* Prevent header from shrinking */
            width: 100%; /* Ensure header spans full width */
        }
        
        .app-title {
            font-size: 24px;
            font-weight: bold;
            margin: 0;
        }
        
        .app-subtitle {
            font-size: 14px;
            opacity: 0.9;
            margin: 0;
        }

        /* Target the parent container of the columns to manage its overall styling */
        .main > div > div.st-emotion-cache-z5fcl4 { /* This is often the div wrapping st.columns, might change */
            padding: 20px; /* Overall padding around the columns content */
            margin: 0;
            width: 100%;
            max-width: 100%;
            flex-grow: 1; /* Allow columns to grow and take available space */
            display: flex; /* Make it a flex container for the columns */
            gap: 20px; /* Gap between columns */
        }

        /* The individual column blocks */
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] { /* This targets the actual column content wrapper */
            flex-grow: 1; /* Each column grows */
            background-color: var(--content-bg) !important;
            color: var(--content-text-color) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--border-radius) !important;
            padding: 20px !important;
            box-shadow: var(--shadow-subtle);
            min-height: 400px; /* Minimum height for columns */
            display: flex; /* Make columns a flex container for their internal content */
            flex-direction: column;
        }

        /* --- Left Column - Upload Section --- */
        .file-item {
            background-color: var(--file-item-bg); /* Theme-dependent file item background */
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 10px 15px;
            margin: 5px 0;
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--content-text-color);
            box-shadow: var(--shadow-subtle);
        }
        
        .file-icon {
            color: var(--st-primary-color); /* Still use Streamlit's primary color */
            font-size: 16px;
        }
        
        /* --- Right Column - Chat Section --- */
        /* Chat history container - make it scrollable */
        /* Targets the specific block that contains chat messages within the chat column */
        div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stChatMessage"]) {
            flex-grow: 1; /* Allow history to take all available space */
            overflow-y: auto; /* Enable scrolling for chat messages */
            padding-right: 10px; /* Space for scrollbar */
            margin-bottom: 10px; /* Space before the input box */
        }
        
        .chat-header {
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 15px;
            margin-bottom: 20px;
            flex-shrink: 0; /* Prevent header from shrinking */
        }
        
        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
            flex-shrink: 0;
        }
        
        /* Status indicator colors - keep these consistent across themes */
        .status-ready { background-color: #d4edda; color: #155724; } /* Light green */
        .status-waiting { background-color: #fff3cd; color: #856404; } /* Light yellow */
        
        /* --- General UI Element Styling --- */
        
        /* Button styling */
        .stButton > button {
            background: var(--primary-gradient);
            color: white !important; /* Force white text */
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            font-weight: 500;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: var(--shadow-medium);
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-strong);
        }
        
        /* Text input and textarea styling */
        .stTextArea textarea, 
        .stTextInput input,
        .stFileUploader span[data-testid="stFileUploadDropzone"],
        .stChatInput > div > div > textarea {
            border-radius: var(--border-radius);
            border: 1px solid var(--border-color); /* Theme-dependent border */
            padding: 12px 20px;
            background-color: var(--component-bg-color); /* Use Streamlit's default component background */
            color: var(--content-text-color);
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.05); /* Inner shadow for depth */
        }
        /* Ensure placeholders are visible */
        .stTextArea textarea::placeholder, 
        .stTextInput input::placeholder,
        .stChatInput > div > div > textarea::placeholder {
            color: var(--st-secondary-text);
            opacity: 0.7;
        }

        /* Streamlit's `st.markdown` for regular text. Ensure it adapts */
        .stMarkdown {
            color: var(--content-text-color); /* General markdown text adapts to theme */
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            color: var(--content-text-color); /* Headers within markdown adapt */
        }

        /* Headers defined in app.py */
        h1, h2, h3, h4, h5, h6 { 
            color: var(--content-text-color); /* Ensure all headers adapt to theme */
            font-weight: 600;
            margin: 0 0 15px 0;
        }
        
        /* --- Chat Message Styling --- */
        div[data-testid="stChatMessage"] {
            background-color: var(--content-bg); /* Fallback, specific styles below */
            color: var(--content-text-color);
            border-radius: var(--border-radius);
            padding: 10px 15px;
            margin-bottom: 10px;
            box-shadow: var(--shadow-subtle);
        }
        
        /* Assistant message specific styling */
        div[data-testid="stChatMessage"]:nth-of-type(odd) { /* Odd children in chat history are usually assistant */
            background-color: var(--chat-bg-assistant); 
            border: 1px solid var(--border-color);
        }
        
        /* User message specific styling */
        div[data-testid="stChatMessage"]:nth-of-type(even) { /* Even children in chat history are usually user */
            background-color: var(--chat-bg-user); 
            border: 1px solid var(--border-color);
        }

        /* Ensure markdown elements inside chat messages also respect the content-text-color */
        div[data-testid="stChatMessage"] .stMarkdown {
            color: var(--content-text-color) !important;
        }

        /* --- Chat Input and Global Controls Styling --- */
        /* The st.chat_input component itself */
        div[data-testid="stChatInput"] {
            background-color: var(--content-bg) !important;
            padding: 10px 0px 0px 0px !important; /* Adjust padding as it's inside a column */
            margin: 0 !important;
            flex-shrink: 0; /* Prevent chat input from shrinking */
        }

        /* Container for clear chat/clear all data buttons */
        .bottom-buttons-container {
            padding: 10px 0px 0px 0px; /* Adjust padding as it's inside a column */
            background-color: var(--content-bg);
            display: flex;
            gap: 15px;
            flex-shrink: 0;
        }
        .bottom-buttons-container > div { /* Target the columns within this container */
            flex: 1;
        }
        
        /* Footer Styling */
        .app-footer {
            text-align: center;
            color: var(--st-secondary-text); /* Use a more subtle color for footer text */
            font-size: 12px;
            padding: 10px;
            flex-shrink: 0;
            background-color: var(--content-bg); /* Footer background adapts */
            border-top: 1px solid var(--border-color);
            width: 100%;
        }

        /* Hide streamlit branding, specific to new versions */
        /* These often appear at the bottom right. */
        .css-1rs6os, .css-17eq0hr, /* Streamlit version 1.25.0+ footer */
        div[data-testid="stDecoration"], /* Small decorative elements */
        div[data-testid="stToolbar"] { /* Toolbar that sometimes appears */
            display: none !important;
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
            .main > div > div.st-emotion-cache-z5fcl4 {
                flex-direction: column; /* Stack columns vertically on smaller screens */
                padding: 10px; /* Reduce padding on small screens */
            }
            div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
                margin-bottom: 20px; /* Add space between stacked columns */
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Gemini client
# @st.cache_resource ensures the client is created once and reused
@st.cache_resource
def get_cached_gemini_client():
    return get_gemini_client()
gemini_sdk = get_cached_gemini_client()

# Initialize in-memory ChromaDB
# @st.cache_resource ensures the client and collection are created once and reused
@st.cache_resource
def get_cached_chroma_setup():
    ch_client_instance = get_chroma_client()
    if not ch_client_instance:
        st.error("Failed to initialize database. Please check your setup.")
        st.stop()
    collection_instance = get_or_create_collection(ch_client_instance, "kb_scout_documents")
    return ch_client_instance, collection_instance
ch_client, collection = get_cached_chroma_setup()


# Session state
if "history" not in st.session_state:
    st.session_state.history = []
    # Add initial greeting message
    st.session_state.history.append({"role": "assistant", "content": "👋 Hello! I'm **K&B Scout AI**, your enterprise document assistant.\nI can help you find information from your uploaded files, or answer general questions. What would you like to know?"})

if "collection" not in st.session_state:
    st.session_state.collection = collection
if "ch_client" not in st.session_state:
    st.session_state.ch_client = ch_client


# ---- App Layout Structure ----

# Custom Header (outside main content block to span full width)
st.markdown(
    """
    <div class="app-header">
        <div style="font-size: 28px;">🤖</div>
        <div>
            <div class="app-title">K&B Scout AI</div>
            <div class="app-subtitle">Enterprise Assistant</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Main Content Area (two columns)
with st.container():
    col1, col2 = st.columns([1, 1.2])

    # Left column - File Upload
    with col1:
        with st.container(): 
            st.markdown("### Upload your files (Temporary Storage)")
            st.markdown("Drag & drop or click to browse. Files are reset on page reload.")
            
            uploaded_files = st.file_uploader(
                "",
                type=["pdf", "csv", "xlsx", "xls", "txt"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            
            st.markdown("**Supports:** .txt, .xls, .xlsx, .csv, .pdf")
            
            if uploaded_files:
                st.markdown(f"**Selected files ({len(uploaded_files)}):**")
                for file in uploaded_files:
                    file_type_icon = {
                        "pdf": "📄", "csv": "📊", "xlsx": "📊", "xls": "📊", 
                        "txt": "📝"
                    }.get(file.name.split('.')[-1].lower(), "📎")
                    
                    st.markdown(
                        f"""
                        <div class="file-item">
                            <span class="file-icon">{file_type_icon}</span>
                            <span>{file.name}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            if uploaded_files:
                if st.button("🚀 Process Files"):
                    rag_chunks: List[RAGChunk] = []
                    
                    with st.status("Processing your files…", expanded=True) as status:
                        total_files = len(uploaded_files)
                        
                        for file_idx, file in enumerate(uploaded_files, 1):
                            st.write(f"Reading **{file.name}** ({file_idx}/{total_files})...")
                            
                            try:
                                file_extension = file.name.split('.')[-1].lower()
                                if file_extension == "pdf":
                                    units = read_pdf(file)
                                elif file_extension == "csv":
                                    units = read_csv(file)
                                elif file_extension in ("xlsx", "xls"):
                                    units = read_xlsx(file)
                                elif file_extension == "txt":
                                    units = read_text(file)
                                else:
                                    st.warning(f"Unsupported file type: {file_extension} for {file.name}. Skipping.")
                                    continue
                                
                                st.write(f"Extracted {len(units)} units from {file.name}")
                                
                                for unit_idx, (unit_text, meta) in enumerate(units):
                                    # Chunks the text using the updated function
                                    chunks = chunk_text(unit_text) 
                                    
                                    for chunk_idx, ch in enumerate(chunks):
                                        if ch.strip():
                                            chunk_meta = meta.copy()
                                            chunk_meta["chunk_id"] = chunk_idx + 1
                                            rag_chunks.append(RAGChunk(id=new_uuid(), text=ch, metadata=chunk_meta))
                            
                            except Exception as e:
                                st.error(f"Failed to read {file.name}: {e}")
                                continue
                        
                        if rag_chunks:
                            st.write(f"Adding {len(rag_chunks)} new chunks to temporary database...")
                            success = add_chunks_to_collection(st.session_state.collection, rag_chunks)
                            
                            if success:
                                status.update(label="✅ Files processed and added to temporary memory", state="complete")
                            else:
                                status.update(label="❌ Failed to process files", state="error")
                        else:
                            status.update(label="ℹ️ No new content to add", state="complete")
            
            st.markdown("---")
            st.markdown("### Currently Loaded Files")
            
            uploaded_files_list = get_uploaded_files_from_collection(st.session_state.collection)
            
            if uploaded_files_list:
                st.markdown(f"**{len(uploaded_files_list)} file(s) in memory:**")
                for filename, filetype in uploaded_files_list:
                    file_icon = {
                        "pdf": "📄", "csv": "📊", "xlsx": "📊", "xls": "📊", 
                        "txt": "📝"
                    }.get(filetype, "📎")
                    
                    st.markdown(
                        f"""
                        <div class="file-item">
                            <span class="file-icon">{file_icon}</span>
                            <span>{filename}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info("No files uploaded yet to current session.")
        
    # Right column - Chat Interface
    with col2:
        with st.container():
            st.markdown('<div class="chat-header">', unsafe_allow_html=True)
            st.markdown("### Chat with K&B Scout AI")
            st.markdown("Ask questions about your uploaded documents or general topics!")
            
            if st.session_state.collection:
                try:
                    count = st.session_state.collection.count()
                    if count > 0:
                        st.markdown(
                            f"""
                            <div class="status-indicator status-ready">
                                🟢 Ready • {count} document chunks indexed (temporary)
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            """
                            <div class="status-indicator status-waiting">
                                🟡 Upload files to get started with document-based answers
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                except Exception as e:
                    st.warning(f"Could not retrieve database status: {e}")
                    st.markdown(
                        """
                        <div class="status-indicator status-waiting">
                            🟡 Database not ready
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            st.markdown('</div>', unsafe_allow_html=True)

            for msg in st.session_state.history:
                avatar = "🤖" if msg["role"] == "assistant" else "👤"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])
            
            st.markdown('<div class="chat-input-area-in-column">', unsafe_allow_html=True)
            prompt = st.chat_input("Ask K&B Scout AI about your documents or anything else...", key="chat_input_col2")

            if prompt:
                st.session_state.history.append({"role": "user", "content": prompt})
                with st.chat_message("user", avatar="👤"):
                    st.markdown(prompt)

                with st.chat_message("assistant", avatar="🤖"):
                    placeholder = st.empty()
                    
                    context_text = ""
                    try:
                        doc_count = st.session_state.collection.count()
                        if doc_count > 0:
                            retrieved = retrieve(st.session_state.collection, prompt)
                            if retrieved:
                                context_text = format_context(retrieved)
                    except Exception as e:
                        st.warning(f"Error during document retrieval: {e}. AI will rely on general knowledge.")
                        context_text = ""

                    try:
                        stream = answer_with_rag(prompt, context_text)
                        answer_accum = ""
                        for chunk in stream:
                            if chunk.candidates[0].content.parts:
                                for part in chunk.candidates[0].content.parts:
                                    if part.text:
                                        answer_accum += part.text
                                        placeholder.markdown(answer_accum + "▌")
                        placeholder.markdown(answer_accum)
                        st.session_state.history.append({"role": "assistant", "content": answer_accum})
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
                        st.session_state.history.pop()

            st.markdown('<div class="bottom-buttons-container-in-column">', unsafe_allow_html=True)
            col_a_chat, col_b_chat = st.columns(2)
            with col_a_chat:
                if st.button("🔄 Clear Chat", key="clear_chat_col2"):
                    st.session_state.history = [{"role": "assistant", "content": "👋 Hello! I'm **K&B Scout AI**, your enterprise document assistant.\nI can help you find information from your uploaded files, or answer general questions. What would you like to know?"}]
                    st.rerun()
            with col_b_chat:
                if st.button("🗑️ Clear All Data (and Reload)", key="clear_all_data_col2"):
                    st.success("All data (and chat) cleared. Reloading page...")
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="app-footer">', unsafe_allow_html=True)
st.markdown(
    """
    💡 <strong>Tip:</strong> Upload your documents on the left, then ask questions about them on the right!<br>
    Your uploaded files are stored **temporarily in memory** and will be reset upon page reload.
    """,
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)