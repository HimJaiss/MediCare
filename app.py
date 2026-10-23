import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Medical Report Assistant",
    page_icon="🩺",
    layout="centered"
)

# =========================
# LOAD ENV
# =========================

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY not found in .env file")
    st.stop()

# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.title("🩺 Medical Assistant")

    st.markdown("""
    Ask questions from patient reports.

    ### Examples
    - What is the diagnosis?
    - What is the blood sugar level?
    - What medicines are mentioned?
    - What abnormalities are reported?
    - Summarize the report
    """)

# =========================
# HEADER
# =========================

st.title("🩺 Medical Report Assistant")
st.caption("AI-powered medical report question answering")

# =========================
# SESSION HISTORY
# =========================

if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# EMBEDDINGS
# =========================

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# =========================
# CHROMA
# =========================

@st.cache_resource
def load_vectorstore():
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

vectorstore = load_vectorstore()

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

doc_count = vectorstore._collection.count()

# =========================
# STATUS CARD
# =========================

st.info(f"📄 Reports Indexed: {doc_count}")

# =========================
# LLM
# =========================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.3,
    max_retries=5
)

# =========================
# PROMPT
# =========================

prompt = ChatPromptTemplate.from_template(
    """
You are a medical report assistant.

Your job is to analyze the patient's medical report and answer the user's question.

IMPORTANT RULES:

1. Use the medical report context as the primary source.
2. Do NOT invent patient information.
3. Do NOT make a diagnosis unless stated in the report.
4. Do NOT prescribe medicines.
5. If the answer is not available in the reports, say:

"I could not find enough information in the reports to answer this question."

Medical Report Context:
{context}

User Question:
{question}

Answer:
"""
)

chain = prompt | llm | StrOutputParser()

# =========================
# EXAMPLE QUESTIONS
# =========================

st.subheader("💡 Quick Questions")

col1, col2 = st.columns(2)

with col1:
    if st.button("Diagnosis"):
        st.session_state.query = "What is the diagnosis?"

    if st.button("Blood Sugar"):
        st.session_state.query = "What is the blood sugar level?"

with col2:
    if st.button("Medicines"):
        st.session_state.query = "What medicines are mentioned?"

    if st.button("Summary"):
        st.session_state.query = "Summarize the report."

# =========================
# QUERY FORM
# =========================

with st.form("question_form", clear_on_submit=False):

    query = st.text_input(
        "Ask a Question",
        value=st.session_state.get("query", ""),
        placeholder="Example: What is the patient's diagnosis?"
    )

    submitted = st.form_submit_button(
        "🔍 Ask Question",
        use_container_width=True
    )

# =========================
# PROCESS QUESTION
# =========================

if submitted:

    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    try:

        with st.spinner("🔎 Searching medical reports..."):

            docs = retriever.invoke(query)

            if not docs:
                st.warning("No relevant report data found.")
                st.stop()

            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            answer = chain.invoke(
                {
                    "context": context,
                    "question": query
                }
            )

        st.session_state.history.insert(
            0,
            {
                "question": query,
                "answer": answer
            }
        )

        st.success("Answer Generated")

        st.subheader("🤖 Answer")

        st.write(answer)

        st.divider()

        st.subheader("📚 Supporting Report Sections")

        for i, doc in enumerate(docs, start=1):

            with st.expander(f"Report Chunk {i}"):

                st.write(doc.page_content)

                if doc.metadata:
                    st.json(doc.metadata)

    except Exception as e:

        st.error("Something went wrong.")
        st.exception(e)

# =========================
# HISTORY
# =========================

if st.session_state.history:

    st.divider()

    st.subheader("🕒 Recent Questions")

    for item in st.session_state.history[:5]:

        with st.expander(item["question"]):

            st.write(item["answer"])