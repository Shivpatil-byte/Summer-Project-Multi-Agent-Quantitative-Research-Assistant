import streamlit as st
import time

# Import your live backend
from rag_backend import get_hybrid_answer

# Initialize the retriever safely from your backend
try:
    from rag_backend import active_retriever as live_retriever
except ImportError:
    st.error("Failed to load backend modules. Please check rag_backend.py")
    live_retriever = None

# ==========================================
# 1. PAGE CONFIG & CUSTOM CSS (BADGES)
# ==========================================
st.set_page_config(page_title="Quant Research Assistant", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .badge-green { background-color: #d4edda; color: #155724; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-amber { background-color: #fff3cd; color: #856404; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-red { background-color: #f8d7da; color: #721c24; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .citation { font-size: 0.85em; color: #555; background: #f0f2f6; padding: 2px 6px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

def render_confidence_badge(score):
    if score >= 0.8:
        return f'<span class="badge-green">Confidence: {score:.2f}</span>'
    elif score >= 0.6:
        return f'<span class="badge-amber">Confidence: {score:.2f}</span>'
    else:
        return f'<span class="badge-red">Confidence: {score:.2f}</span>'

# ==========================================
# 2. SESSION STATE MANAGEMENT (MEMORY)
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "latest_chunks" not in st.session_state:
    st.session_state.latest_chunks = []

# ==========================================
# 3. SIDEBAR: RETRIEVAL DIAGNOSTICS
# ==========================================
with st.sidebar:
    st.header("🔍 Retrieval Diagnostics")
    st.markdown("Expand to view chunks retrieved by the Hybrid (RRF + Cross-Encoder) pipeline.")
    
    if st.session_state.latest_chunks:
        for idx, chunk in enumerate(st.session_state.latest_chunks):
            with st.expander(f"Chunk {idx + 1} (Score: {chunk['score']:.2f})"):
                st.markdown(f"**Source:** {chunk['document']} ({chunk['year']}), Page {chunk['page']}")
                st.write(chunk['text'])
    else:
        st.info("No chunks retrieved yet. Ask a question to begin.")

# ==========================================
# 4. MAIN CHAT INTERFACE
# ==========================================
st.title("📈 Multi-Agent Quantitative Research Assistant")
st.markdown("Ask financial or quantitative questions based on your ingested reports.")

# Render existing chat history on page load
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("E.g., What was the exact operating margin for Danone's water segment in 2023?"):
    
    # 1. Add user message to state and UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # 2. Enforce Memory Limit (Rolling window of last 5 turns)
    if len(st.session_state.messages) > 10:
        st.session_state.messages = st.session_state.messages[-10:]

    # 3. Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing financial reports (Hybrid Retrieval & Re-ranking)..."):
            
            if live_retriever is None:
                st.error("Retriever is not initialized. Please load your VectorStore and BM25 index in the backend script.")
                st.stop()
            else:
                # Execution of the live pipeline
                real_response = get_hybrid_answer(prompt, st.session_state.messages, live_retriever)
            
        # 4. Format the output FIRST
        citations_html = " ".join([f'<span class="citation">[{c["doc"]}, {c["year"]}, p. {c["page"]}]</span>' for c in real_response["citations"]])
        badge_html = render_confidence_badge(real_response["confidence"])
        full_output = f"{real_response['answer']}<br><br>{citations_html}<br><br>{badge_html}"
        
        # 5. SAVE to memory and state BEFORE rerunning
        st.session_state.messages.append({"role": "assistant", "content": full_output})
        st.session_state.latest_chunks = real_response["retrieved_chunks"]
        
        # 6. Now safely refresh the UI to show the new chat and sidebar
        st.rerun()