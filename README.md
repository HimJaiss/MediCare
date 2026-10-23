
# MediCare — AI-Powered Medical Report Analysis & Question Answering System

MediCare is a Python-based medical document question-answering application built using Retrieval-Augmented Generation (RAG). It processes medical reports in PDF format, converts document text into vector embeddings, retrieves relevant information using semantic similarity search, and uses Google Gemini AI to generate context-aware answers grounded in the provided documents.

The system combines document processing, vector search, and Large Language Models (LLMs) to help users find relevant information from medical reports through an interactive web interface.

## Key Features

- **Medical PDF Processing:** Extract text from medical reports in PDF format using LangChain document loaders.
- **Intelligent Document Chunking:** Split extracted text into smaller, overlapping chunks to improve retrieval relevance and preserve context.
- **Semantic Search:** Generate document embeddings using the `sentence-transformers/all-MiniLM-L6-v2` model and retrieve relevant report sections.
- **RAG-Based Question Answering:** Combine retrieved document context with Google Gemini AI to generate context-aware answers.
- **Vector Database:** Store and retrieve document embeddings using ChromaDB for semantic similarity search.
- **Interactive Web Interface:** Use Streamlit to ask questions and view generated answers alongside supporting report sections.
- **Source Transparency:** Display relevant document excerpts and metadata to help users identify the information used to answer a question.
- **Environment-Based Configuration:** Store API credentials in a local `.env` file instead of hardcoding secrets.
- **Context-Aware Responses:** Indicate when the available reports do not contain enough information to answer a question.

## Tech Stack

| Domain | Technology / Library |
|---|---|
| Programming Language | Python |
| Web Interface | Streamlit |
| RAG Framework | LangChain |
| Large Language Model | Google Gemini API |
| Embeddings | Hugging Face Sentence Transformers |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database | ChromaDB |
| PDF Processing | PyPDFLoader, pypdf |
| Document Chunking | RecursiveCharacterTextSplitter |
| Environment Configuration | python-dotenv |
| Version Control | Git and GitHub |

## Project Structure

```text
MediCare/
├── app.py                  # Streamlit application and RAG interface
├── train.py                # Loads PDF reports and builds the vector database
├── ask.py                  # Retrieves relevant document chunks for questions
├── rag_chat.py             # RAG-based conversational question-answering logic
├── train copy.py           # Additional document-processing script
├── patients_reports/       # Input medical PDF reports (local/private)
├── chroma_db/              # Persistent ChromaDB vector database (local)
├── .env                    # Local API credentials (not committed)
├── .gitignore              # Files and folders excluded from Git
├── README.md               # Project documentation
└── .venv/                  # Python virtual environment (not committed)
```

## Getting Started

### Prerequisites

- Python 3.13 or another Python version compatible with the project dependencies
- Git
- pip
- A Google Gemini API key
- Medical reports in text-readable PDF format

### 1. Clone the Repository

Replace `YOUR_GITHUB_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/MediCare.git
cd MediCare
```

### 2. Create and Activate a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install streamlit python-dotenv langchain langchain-community langchain-text-splitters langchain-huggingface langchain-chroma langchain-google-genai sentence-transformers pypdf
```

### 4. Configure Environment Variables

Create a `.env` file in the project root and add your Google Gemini API key:

```dotenv
GOOGLE_API_KEY=your_actual_gemini_api_key
```

Replace the placeholder with your actual API key. Do not commit your `.env` file or expose your API key publicly.

You can obtain an API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 5. Add PDF Reports

Place your medical PDF files inside the `patients_reports/` directory.

```text
patients_reports/
├── sample_medical_report.pdf
└── another_report.pdf
```

Use fictional or properly de-identified documents for development and testing. The sample report created during development contains fictional information and is not intended for diagnosis or treatment.

### 6. Build the Vector Database

Run the document-processing script:

```bash
python train.py
```

The script:

1. Loads PDF files from `patients_reports/`.
2. Extracts their text.
3. Splits the text into chunks of 300 characters with an overlap of 50 characters.
4. Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
5. Stores the document chunks and embeddings in the persistent `chroma_db/` directory.

The first run may take some time because the embedding model needs to be downloaded.

**Note:** The current training script creates a Chroma database from the supplied documents. If you add reports or rebuild the database, manage existing data carefully to avoid duplicate or stale entries.

### 7. Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually:

```text
http://localhost:8501
```

Keep the terminal open while using the application.

## System Architecture

```text
             [ Medical PDF Reports ]
                       |
                       v
             [ PDF Text Extraction ]
                       |
                       v
               [ Document Chunking ]
                 Chunk Size: 300
                 Overlap: 50
                       |
                       v
             [ Embedding Generation ]
             Sentence Transformers
                       |
                       v
             [ ChromaDB Vector Store ]
                       |
                       v
              [ User Question ]
                       |
                       v
                [ Semantic Search ]
                       |
                       v
            [ Retrieved Document Context ]
                       |
                       v
               [ Google Gemini AI ]
                       |
                       v
            [ Generated Answer + Sources ]
```

## How the RAG Pipeline Works

1. **Document Ingestion:** PDF reports are loaded and their text is extracted.
2. **Chunking:** Extracted text is divided into smaller, overlapping sections.
3. **Embedding Generation:** Each chunk is converted into a numerical vector using the Sentence Transformers embedding model.
4. **Vector Storage:** The embeddings and associated document metadata are stored in ChromaDB.
5. **Question Retrieval:** When a user asks a question, the application retrieves relevant document sections using semantic similarity search.
6. **Context Construction:** The retrieved sections are provided to the language model as context.
7. **Answer Generation:** Google Gemini generates an answer using the supplied context.
8. **Source Display:** The Streamlit interface displays the answer and supporting report sections.

## Example Use Case

**Question:**

What temperature is listed in the report?

**Relevant Report Information:**

Temperature: 36.8 degrees Celsius.

**Expected Answer:**

The report lists a temperature of 36.8°C.

If a report does not contain the requested information, such as a documented diagnosis, the application should indicate that there is insufficient information rather than inventing a medical conclusion.

## Security and Privacy

- Keep API keys in the local `.env` file.
- Exclude `.env`, `.venv/`, and `chroma_db/` from version control.
- Do not commit confidential medical reports or personally identifiable patient information.
- Use fictional or properly de-identified documents for development and testing.
- Review access permissions and data-handling requirements before using sensitive documents.

## Future Improvements

- Add PDF uploads directly through the Streamlit interface.
- Support selecting and filtering individual reports.
- Implement hybrid search and reranking for improved retrieval quality.
- Add automated evaluation of retrieval relevance and answer accuracy.
- Support incremental indexing when new documents are added.
- Improve error handling for missing API keys, empty PDFs, and unsupported documents.
- Introduce access controls and stronger privacy protections for sensitive medical information.

## Disclaimer

MediCare is an educational document-retrieval and question-answering project. It is not a medical diagnostic system and must not replace professional medical advice, diagnosis, or treatment. AI-generated answers may be incomplete or inaccurate. Always verify important information against the original report and consult a qualified healthcare professional for medical decisions.

## License

This project is intended for educational and demonstration purposes. Add an appropriate open-source license if you intend to permit reuse, modification, and redistribution.
