from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Same embedding model jo train.py me use kiya tha
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Existing ChromaDB load karo
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# User se question lo
query = input("Ask Question: ")

# Top 3 relevant chunks retrieve karo
results = vectorstore.similarity_search(query, k=3)

print("\n" + "=" * 50)
print("RETRIEVED CHUNKS")
print("=" * 50)

for i, doc in enumerate(results, start=1):
    print(f"\nChunk {i}")
    print("-" * 50)
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)