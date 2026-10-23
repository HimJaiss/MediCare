from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load Environment Variables
load_dotenv()

# ==========================
# STEP 1 - LOAD PDF FILES
# ==========================

loader = DirectoryLoader(
    "patients_reports",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()

print(f"Total Documents Loaded: {len(docs)}")


# ==========================
# STEP 2 - CHUNKING
# ==========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

print(f"Total Chunks Created: {len(chunks)}")

print("\nSample Metadata:\n")
print(chunks[0].metadata)


# ==========================
# STEP 3 - EMBEDDINGS
# ==========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding Model Loaded")


# ==========================
# STEP 4 - VECTOR DATABASE
# ==========================

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("Vector Database Created Successfully!")