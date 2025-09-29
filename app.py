import streamlit as st
from typing import List
import contextlib

# Suppress errors at import time
with contextlib.suppress(Exception):
    from transformers import pipeline
with contextlib.suppress(Exception):
    import spacy
with contextlib.suppress(Exception):
    from spacy.cli import download as spacy_download


# ---------------------------
# Helpers
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_summarizer():
    try:
        return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    except Exception:
        try:
            return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
        except Exception as exc:
            st.error("❌ Failed to load summarizer model. Check your internet connection.")
            raise exc


@st.cache_resource(show_spinner=False)
def get_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        try:
            spacy_download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
        except Exception:
            st.warning("⚠️ Could not download spaCy model. Entity recognition disabled.")
            return None


def chunk_text_by_words(text: str, max_words: int = 800, overlap: int = 50) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def summarize_long_text(text: str) -> str:
    summarizer = get_summarizer()
    chunks = chunk_text_by_words(text, max_words=800, overlap=50)
    if not chunks:
        return ""
    if len(chunks) == 1:
        return summarizer(chunks[0], max_length=130, min_length=50, do_sample=False)[0]["summary_text"]

    partial_summaries = []
    progress = st.progress(0.0, text="🔄 Summarizing chunks…")
    for idx, chunk in enumerate(chunks, start=1):
        summary = summarizer(chunk, max_length=130, min_length=50, do_sample=False)[0]["summary_text"]
        partial_summaries.append(summary)
        progress.progress(idx / len(chunks), text=f"Chunk {idx}/{len(chunks)} summarized")
    progress.empty()

    combined = " ".join(partial_summaries)
    final = summarizer(combined, max_length=160, min_length=60, do_sample=False)[0]["summary_text"]
    return final


def extract_entities_and_actions(text: str):
    nlp = get_nlp()
    if nlp is None:
        return {"deadlines": [], "people": [], "action_items": []}

    doc = nlp(text)
    deadlines = [ent.text for ent in doc.ents if ent.label_ in ["DATE", "TIME"]]
    people = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    action_items = []
    trigger_words = ["will", "must", "need to", "should", "action", "todo", "follow up"]
    for sent in doc.sents:
        if any(trigger in sent.text.lower() for trigger in trigger_words):
            action_items.append(sent.text.strip())

    return {"deadlines": deadlines, "people": people, "action_items": action_items}


def analyze_notes(text: str):
    return {"summary": summarize_long_text(text), **extract_entities_and_actions(text)}


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI Meeting Assistant", layout="wide")


st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/4712/4712107.png",
    width=80
)

st.sidebar.title("🤖 Meeting Assistant")
st.sidebar.write("Your AI agent for summarizing notes and extracting key insights.")
st.sidebar.markdown("---")
st.sidebar.write("Built by **Muhib**")
st.sidebar.markdown("[Portfolio](https://syedmuhib.vercel.app)")


# Main Layout
st.title("📋 AI Meeting Assistant Agent")
st.write("Paste your meeting notes below and get a **smart summary, deadlines, people, and action items** instantly.")

user_text = st.text_area("✍️ Paste Meeting Notes Here", height=200, placeholder="Paste your meeting notes or transcript…")


col1, col2 = st.columns([1, 1])
with col1:
    run_clicked = st.button("🚀 Analyze Notes", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("🧹 Clear", use_container_width=True)
if clear_clicked:
    st.session_state.meeting_notes = ""  # clears the text box
    st.rerun()
if run_clicked:
    if user_text.strip():
        with st.spinner("⚡ Analyzing notes... please wait."):
            result = analyze_notes(user_text)

        # Summary Card
        with st.container():
            st.markdown("### ✅ Summary")
            st.info(result.get("summary") or "No summary generated.")

        # Deadlines Card
        with st.container():
            st.markdown("### 📅 Deadlines")
            deadlines = result.get("deadlines", [])
            if deadlines:
                for d in dict.fromkeys(deadlines):
                    st.markdown(f"- 🟢 **{d}**")
            else:
                st.write("No deadlines found.")

        # People Card
        with st.container():
            st.markdown("### 👤 People")
            people = result.get("people", [])
            if people:
                for p in dict.fromkeys(people):
                    st.markdown(f"- 🔵 **{p}**")
            else:
                st.write("No people found.")

        # Action Items Card
        with st.container():
            st.markdown("### 📝 Action Items")
            action_items = result.get("action_items", [])
            if action_items:
                for item in dict.fromkeys(action_items):
                    st.markdown(f"- 🟠 {item}")
            else:
                st.write("No action items found.")
    else:
        st.warning("⚠️ Please paste some text first.")
