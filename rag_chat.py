from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# ==========================
# Embeddings
# ==========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================
# Load Chroma
# ==========================

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# ==========================
# Retriever
# ==========================

retriever = vectorstore.as_retriever(
    search_kwargs={"k":3}
)


# ==========================
# Gemini
# ==========================

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=1.5, max_retries=5)

# ==========================
# Prompt
# ==========================

prompt = ChatPromptTemplate.from_template(
"""
You are a medical report assistant.

Your job is to analyze the patient's medical report and answer the user's question.

IMPORTANT RULES:

1. Use the medical report context as the primary source for the patient's
   actual test results and findings.

2. If the user asks for precautions, lifestyle suggestions, diet suggestions,
   or what they should do about an abnormal result, you may provide GENERAL
   health guidance based on the findings in the report.

3. Do NOT invent patient information that is not present in the report.

4. Do NOT make a definite diagnosis unless the diagnosis is explicitly stated
   in the report.

5. Do NOT prescribe medicines or give specific medication doses.

6. If a test result is abnormal, explain:
   - What the result shows
   - Possible general implications
   - General precautions
   - When the patient should discuss it with a doctor

7. Clearly distinguish between information found in the report and general
   health guidance.

8. If the report does not contain enough information to answer the question,
   say and if it is a personal question say:
   "I could not find enough information in the reports to answer this question."
    else it is a general question try to answer by your parametric knowledgde.: 

Medical Report Context:
{context}

User Question:
{question}

Answer:
"""
)


# ==========================
# RAG Chain
# ==========================

def get_context(question):
    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return context



rag_chain = (
    {
        "context": get_context,
        "question": RunnablePassthrough()
    }
    |
    prompt
    |
    llm
    |
    StrOutputParser()
)


# ==========================
# Chat
# ==========================

while True:

    query = input("\nAsk Question: ")

    if query.lower()=="exit":
        break


    response = rag_chain.invoke(query)

    print("\nAnswer:")
    print(response)