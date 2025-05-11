# assignment6
# RAG System Implementation

This repository contains a complete implementation of a Retrieval-Augmented Generation (RAG) system using open-source components with local vector storage, as required for CSAI 422 Laboratory Assignment 6.

## Overview

The RAG system enhances Large Language Model (LLM) capabilities by retrieving relevant information from external knowledge sources before generating responses. This implementation includes:

- Document processing with support for PDF, DOCX, and TXT files
- Document chunking with customizable size and overlap
- Multiple embedding models using Sentence Transformers
- Efficient vector storage with FAISS
- Various retrieval strategies (similarity search, MMR, hybrid search)
- Integration with the Groq API for LLM generation
- Comprehensive evaluation framework

## Setup Instructions

### Prerequisites

- Python 3.10+
- Groq API Key

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/rag-system.git
   cd rag-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. Create a `documents` folder in the project root and add your documents (PDF, DOCX, TXT):
   ```bash
   mkdir documents
   # Copy your documents to the documents folder
   ```

## Usage

### Running the Example Script

```bash
python rag_system_run.py
```

This will start a menu where you can choose to run example usage or interactive mode.

### Using the RAG System in Your Code

```python
from rag_system_implementation import RAGSystem

# Create a RAG system
rag_system = RAGSystem()

# Process documents
rag_system.process_documents(chunk_size=1000, chunk_overlap=200)

# Query the system
response, retrieved_docs = rag_system.query(
    query="What is RAG?",
    k=4,
    retrieval_strategy="mmr",
    template_type="standard"
)

print(response)
```

## System Architecture

The RAG system consists of the following components:

### 1. Document Processor

- Loads documents from various file formats (PDF, DOCX, TXT)
- Splits documents into chunks using RecursiveCharacterTextSplitter
- Preserves document metadata throughout processing

### 2. Embedding Manager

- Generates embeddings using Sentence Transformers
- Supports multiple embedding models:
  - `all-MiniLM-L6-v2` (default)
  - `all-mpnet-base-v2`

### 3. Vector Store

- Stores document embeddings using FAISS for efficient similarity search
- Implements various retrieval strategies:
  - Basic similarity search
  - Maximum Marginal Relevance (MMR)
  - Hybrid search (combining semantic and keyword search)
- Supports saving and loading vector stores to/from disk

### 4. RAG System

- Integrates document processing, embedding, retrieval, and generation
- Implements prompt templates for different query types
- Provides interface for querying the system

### 5. RAG Evaluator

- Evaluates retrieval performance (precision, recall, F1)
- Assesses answer quality using the LLM as a judge
- Compares different RAG configurations

## Retrieval Strategies

### 1. Basic Similarity Search

- Retrieves documents based on cosine similarity
- Fast and efficient for most use cases

### 2. Maximum Marginal Relevance (MMR)

- Balances relevance and diversity in search results
- Useful when you want to avoid redundant information

### 3. Hybrid Search

- Combines semantic search with keyword-based search
- Particularly effective for queries with specific terms

## Advanced RAG Techniques

### 1. Query Rewriting

- Uses the LLM to rewrite the user's query for better retrieval
- Helps with ambiguous or underspecified queries

### 2. Post-Retrieval Filtering

- Removes irrelevant documents after retrieval
- Improves precision of the retrieved documents

### 3. Hierarchical Retrieval

- Two-stage retrieval process with coarse and fine-grained steps
- Helps with scaling to larger document collections

### 4. Self-Correcting RAG

- Evaluates and potentially regenerates answers
- Improves answer quality through self-reflection

## Evaluation Results

The system evaluates different configurations using:

- Retrieval metrics (precision, recall, F1)
- Answer quality assessment

Results are visualized to show the performance of different:
- Embedding models
- Retrieval strategies
- Prompt templates

## Strengths and Weaknesses

### Strengths

- Modular design that allows easy swapping of components
- Support for multiple retrieval strategies
- Comprehensive evaluation framework
- Advanced RAG techniques for improved performance
- Good documentation and example usage

### Weaknesses

- Limited to text-based documents
- Relies on pre-trained embedding models
- Vector search might not work well for all types of queries
- Evaluation is partially subjective when using LLM as judge

## Future Improvements

- Add support for more document types (HTML, JSON, etc.)
- Implement more embedding models
- Add support for hybrid retrieval with other algorithms
- Implement caching for improved performance
- Add support for reranking retrieved documents

## Requirements

- langchain
- faiss-cpu
- sentence-transformers
- groq (for API integration)
- python-dotenv
- matplotlib
- numpy
- scikit-learn
- PyPDF2
- docx2txt
- tqdm

## License

[MIT License](LICENSE)
