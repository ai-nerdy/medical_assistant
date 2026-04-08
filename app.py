import os
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Assistant - RAG",
    page_icon="🏥",
    layout="wide",
)

# ── Sidebar: API credentials ────────────────────────────────
st.sidebar.title("⚙️ Configuration")

# Try loading from Streamlit secrets first, then fall back to sidebar input
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError, Exception):
    openai_api_key = ""
try:
    openai_api_base = st.secrets["OPENAI_API_BASE"]
except (KeyError, FileNotFoundError, Exception):
    openai_api_base = ""

if not openai_api_key:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_base:
    openai_api_base = st.sidebar.text_input("OpenAI API Base URL", value="https://api.openai.com/v1")

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
if openai_api_base:
    os.environ["OPENAI_BASE_URL"] = openai_api_base


# ── Build / cache the vector store ──────────────────────────
@st.cache_resource(show_spinner="📚 Loading & indexing the medical manual… (runs once)")
def build_vectorstore(pdf_path: str, _api_key: str, _api_base: str):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=512,
    )
    chunks = text_splitter.split_documents(documents)

    embedding_model = OpenAIEmbeddings(
        openai_api_key=_api_key,
        openai_api_base=_api_base,
        chunk_size=100,
    )
    vectorstore = Chroma.from_documents(chunks, embedding_model)
    return vectorstore


# ── Prompts (same as your notebook) ─────────────────────────
QNA_SYSTEM_MESSAGE = """
You are an AI assistant designed to support healthcare professionals and patients in efficiently
accessing evidence-based medical knowledge. Your task is to provide concise, accurate, and clinically
relevant answers based on the context provided from trusted medical references such as the Merck Manuals.

User input will include the necessary context for you to answer their questions. This context
will begin with the token:

###Context
The context contains excerpts from one or more medical manuals, along with associated metadata
such as section titles, page numbers, and relevant subsections.

When crafting your response:
- Use only the provided context to answer the question.
- If the answer is found in the context, respond with clear, guideline-driven summaries.
- Include the section title and page number (or subsection reference) as the source.
- If the question is unrelated to the context or the context is empty, clearly respond with:
  "This information is not available in the provided context. For safe and personalized medical
   advice, please consult a licensed healthcare professional."

Please adhere to the following response guidelines:
- Provide clear, direct answers using only the given context.
- Do not include any additional information outside of the context.
- Avoid rephrasing or generalizing unless explicitly relevant to the question.
- If no relevant answer exists in the context, respond with:
  "This information is not available in the provided context. For safe and personalized medical
   advice, please consult a licensed healthcare professional."
- If the context is not provided, your response should also be:
  "This information is not available in the provided context. For safe and personalized medical
   advice, please consult a licensed healthcare professional."

Here is an example of how to structure your response:

Answer:
[Answer based on context]

Source:
[Section title, page number or subsection reference]
"""

QNA_USER_TEMPLATE = """
###Context
{context}

###Question
{question}

Instructions:
Use the context above to answer the question as accurately as possible.
If the context contains the answer, provide a concise, evidence-based summary and cite the source.
If the context is empty or unrelated, respond with:
"This information is not available in the provided context. For safe and personalized medical
advice, please consult a licensed healthcare professional."
"""


def generate_rag_response(client, retriever, user_input, max_tokens=500, temperature=0.3, top_p=0.95):
    relevant_docs = retriever.get_relevant_documents(query=user_input)
    context_for_query = ". ".join([d.page_content for d in relevant_docs])

    user_message = QNA_USER_TEMPLATE.format(context=context_for_query, question=user_input)

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": QNA_SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Sorry, I encountered the following error:\n{e}"


# ── Main UI ──────────────────────────────────────────────────
st.title("🏥 Medical Assistant")
st.markdown("Ask evidence-based medical questions powered by RAG over the **Merck Manual**.")

PDF_PATH = "medical_diagnosis_manual.pdf"

if not os.path.exists(PDF_PATH):
    st.warning(
        f"⚠️ PDF not found at `{PDF_PATH}`. "
        "Please place the **medical_diagnosis_manual.pdf** file in the same directory as this app."
    )
    st.stop()

if not openai_api_key:
    st.info("👈 Enter your OpenAI API key in the sidebar to get started.")
    st.stop()

# Build vector store (cached)
vectorstore = build_vectorstore(PDF_PATH, openai_api_key, openai_api_base)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
client = OpenAI()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Sample questions
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Sample Questions")
sample_questions = [
    "What is the protocol for managing sepsis in a critical care unit?",
    "What are the common symptoms for appendicitis, and can it be cured via medicine?",
    "What are the effective treatments for sudden patchy hair loss?",
    "What treatments are recommended for a physical injury to brain tissue?",
]
for sq in sample_questions:
    if st.sidebar.button(sq, key=sq):
        st.session_state["_prefill"] = sq

# Handle prefilled question from sidebar
prefill = st.session_state.pop("_prefill", None)

# Chat input
user_input = st.chat_input("Ask a medical question…") or prefill

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = generate_rag_response(client, retriever, user_input)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
