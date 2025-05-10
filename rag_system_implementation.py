"""
CSAI 422: Laboratory Assignment 6
RAG System Implementation using Groq API
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

class DocumentProcessor:
    """Class for loading and processing documents."""
    
    def __init__(self, documents_dir: str = "./documents"):
        """Initialize the document processor.
        
        Args:
            documents_dir: Directory containing the documents to process.
        """
        self.documents_dir = Path(documents_dir)
        if not self.documents_dir.exists():
            logger.info(f"Creating documents directory: {self.documents_dir}")
            self.documents_dir.mkdir(parents=True)
    
    def load_documents(self) -> List[Document]:
        """Load documents from the documents directory.
        
        Returns:
            List of loaded documents.
        """
        documents = []
        
        if not list(self.documents_dir.iterdir()):
            logger.warning(f"No documents found in {self.documents_dir}")
            return documents
        
        logger.info(f"Loading documents from {self.documents_dir}")
        
        for file_path in tqdm(list(self.documents_dir.iterdir()), desc="Loading documents"):
            try:
                if file_path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(file_path))
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = file_path.name
                        doc.metadata["file_type"] = "pdf"
                    documents.extend(docs)
                
                elif file_path.suffix.lower() == '.txt':
                    loader = TextLoader(str(file_path))
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = file_path.name
                        doc.metadata["file_type"] = "txt"
                    documents.extend(docs)
                
                elif file_path.suffix.lower() in ['.docx', '.doc']:
                    loader = Docx2txtLoader(str(file_path))
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = file_path.name
                        doc.metadata["file_type"] = "docx"
                    documents.extend(docs)
                
                else:
                    logger.warning(f"Unsupported file format: {file_path}")
            
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def split_documents(self, documents: List[Document], 
                        chunk_size: int = 1000, 
                        chunk_overlap: int = 200) -> List[Document]:
        """Split documents into chunks.
        
        Args:
            documents: List of documents to split.
            chunk_size: Size of each chunk.
            chunk_overlap: Overlap between chunks.
            
        Returns:
            List of document chunks.
        """
        logger.info(f"Splitting documents with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        return chunks


class EmbeddingManager:
    """Class for managing document embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding manager.
        
        Args:
            model_name: Name of the sentence transformer model to use.
        """
        logger.info(f"Initializing embedding model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")
    
    def generate_embeddings(self, documents: List[Document]) -> np.ndarray:
        """Generate embeddings for a list of documents.
        
        Args:
            documents: List of documents to generate embeddings for.
            
        Returns:
            Array of document embeddings.
        """
        texts = [doc.page_content for doc in documents]
        logger.info(f"Generating embeddings for {len(texts)} documents")
        
        embeddings = self.model.encode(texts, show_progress_bar=True)
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        
        return embeddings
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a query.
        
        Args:
            query: Query to generate embedding for.
            
        Returns:
            Query embedding.
        """
        return self.model.encode([query])[0]


class VectorStore:
    """Class for managing vector storage and retrieval."""
    
    def __init__(self, dimension: int):
        """Initialize the vector store.
        
        Args:
            dimension: Dimension of the vectors to store.
        """
        logger.info(f"Initializing FAISS vector store with dimension {dimension}")
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
    
    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents and their embeddings to the vector store.
        
        Args:
            documents: List of documents to add.
            embeddings: Embeddings of the documents.
        """
        logger.info(f"Adding {len(documents)} documents to vector store")
        self.documents.extend(documents)
        self.index.add(embeddings)
        logger.info(f"Vector store now contains {len(self.documents)} documents")
    
    def similarity_search(self, query_embedding: np.ndarray, k: int = 4) -> List[Tuple[Document, float]]:
        """Perform similarity search.
        
        Args:
            query_embedding: Embedding of the query.
            k: Number of results to return.
            
        Returns:
            List of (document, score) tuples.
        """
        if len(self.documents) == 0:
            logger.warning("No documents in vector store")
            return []
        
        k = min(k, len(self.documents))
        
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32), k
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents) and idx >= 0:
                results.append((self.documents[idx], float(distances[0][i])))
        
        return results
    
    def maximum_marginal_relevance_search(
        self, query_embedding: np.ndarray, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5
    ) -> List[Tuple[Document, float]]:
        """Perform Maximum Marginal Relevance search.
        
        Args:
            query_embedding: Embedding of the query.
            k: Number of results to return.
            fetch_k: Number of initial results to fetch.
            lambda_mult: Diversity parameter.
            
        Returns:
            List of (document, score) tuples.
        """
        if len(self.documents) == 0:
            logger.warning("No documents in vector store")
            return []
        
        fetch_k = min(fetch_k, len(self.documents))
        k = min(k, fetch_k)
        
        # Get initial results
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32), fetch_k
        )
        
        # Convert to embeddings
        initial_embeddings = []
        initial_indices = []
        for idx in indices[0]:
            if idx < len(self.documents) and idx >= 0:
                doc_content = self.documents[idx].page_content
                doc_embedding = self.model.encode([doc_content])[0]
                initial_embeddings.append(doc_embedding)
                initial_indices.append(idx)
        
        initial_embeddings = np.array(initial_embeddings)
        
        # Calculate MMR
        selected_indices = []
        selected_indices_set = set()
        
        for _ in range(k):
            if len(selected_indices) == len(initial_indices):
                break
            
            # Calculate relevance scores
            relevance_scores = cosine_similarity(
                [query_embedding], initial_embeddings
            )[0]
            
            # Calculate diversity scores
            if selected_indices:
                selected_embeddings = initial_embeddings[selected_indices]
                diversity_scores = np.max(cosine_similarity(
                    initial_embeddings, selected_embeddings
                ), axis=1)
            else:
                diversity_scores = np.zeros(len(initial_embeddings))
            
            # Calculate MMR scores
            mmr_scores = lambda_mult * relevance_scores - (1 - lambda_mult) * diversity_scores
            
            # Mask already selected
            for idx in selected_indices:
                mmr_scores[idx] = -np.inf
            
            # Select the best
            next_idx = np.argmax(mmr_scores)
            if next_idx not in selected_indices_set:
                selected_indices.append(next_idx)
                selected_indices_set.add(next_idx)
        
        # Get results
        results = []
        for idx in selected_indices:
            original_idx = initial_indices[idx]
            results.append((self.documents[original_idx], float(distances[0][initial_indices.index(original_idx)])))
        
        return results
    
    def hybrid_search(self, query: str, documents: List[Document], k: int = 4, alpha: float = 0.5) -> List[Tuple[Document, float]]:
        """Perform hybrid search combining semantic and keyword-based search.
        
        Args:
            query: Query string.
            documents: List of documents to search.
            k: Number of results to return.
            alpha: Weight of semantic search (vs keyword search).
            
        Returns:
            List of (document, score) tuples.
        """
        # Semantic search
        query_embedding = self.model.encode([query])[0]
        semantic_results = self.similarity_search(query_embedding, k=k)
        
        # Keyword search using TF-IDF
        tfidf = TfidfVectorizer()
        document_texts = [doc.page_content for doc in documents]
        tfidf_matrix = tfidf.fit_transform(document_texts)
        query_tfidf = tfidf.transform([query])
        
        # Calculate cosine similarity
        cosine_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]
        
        # Combine results
        hybrid_scores = {}
        for i, doc in enumerate(documents):
            hybrid_scores[i] = alpha * cosine_scores[i]
        
        for doc, score in semantic_results:
            idx = documents.index(doc)
            hybrid_scores[idx] += (1 - alpha) * (1 / (score + 1))  # Convert distance to similarity
        
        # Sort and return top k
        sorted_indices = sorted(hybrid_scores.keys(), key=lambda idx: hybrid_scores[idx], reverse=True)[:k]
        
        return [(documents[idx], hybrid_scores[idx]) for idx in sorted_indices]
    
    def save(self, file_path: str):
        """Save the vector store to disk.
        
        Args:
            file_path: Path to save the vector store to.
        """
        logger.info(f"Saving vector store to {file_path}")
        faiss.write_index(self.index, f"{file_path}.index")
        
        documents_data = []
        for doc in self.documents:
            doc_dict = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            documents_data.append(doc_dict)
        
        with open(f"{file_path}.json", "w") as f:
            json.dump(documents_data, f)
        
        logger.info(f"Saved vector store with {len(self.documents)} documents")
    
    def load(self, file_path: str):
        """Load the vector store from disk.
        
        Args:
            file_path: Path to load the vector store from.
        """
        logger.info(f"Loading vector store from {file_path}")
        
        if not os.path.exists(f"{file_path}.index") or not os.path.exists(f"{file_path}.json"):
            logger.error(f"Vector store files not found at {file_path}")
            return
        
        self.index = faiss.read_index(f"{file_path}.index")
        
        with open(f"{file_path}.json", "r") as f:
            documents_data = json.load(f)
        
        self.documents = []
        for doc_dict in documents_data:
            doc = Document(
                page_content=doc_dict["page_content"],
                metadata=doc_dict["metadata"]
            )
            self.documents.append(doc)
        
        logger.info(f"Loaded vector store with {len(self.documents)} documents")


class RAGSystem:
    """Class for the RAG system."""
    
    def __init__(self, model_name: str = "llama3-70b-8192"):
        """Initialize the RAG system.
        
        Args:
            model_name: Name of the Groq model to use.
        """
        logger.info(f"Initializing RAG system with model {model_name}")
        
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=model_name
        )
        
        self.document_processor = DocumentProcessor()
        self.embedding_managers = {
            "all-MiniLM-L6-v2": EmbeddingManager("all-MiniLM-L6-v2"),
            "all-mpnet-base-v2": EmbeddingManager("all-mpnet-base-v2")
        }
        
        self.vector_stores = {}
        self.documents = []
        self.chunks = []
        
        # Initialize default embedding manager
        self.current_embedding_manager = self.embedding_managers["all-MiniLM-L6-v2"]
        
        # Set up RAG prompt templates
        self.setup_prompt_templates()
    
    def setup_prompt_templates(self):
        """Set up prompt templates for the RAG system."""
        # Standard prompt template
        self.standard_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions based on the provided context. "
                      "If the answer cannot be found in the context, say that you don't know."),
            ("human", "Context:\n{context}\n\nQuestion: {query}")
        ])
        
        # Specialized template for factual questions
        self.factual_template = ChatPromptTemplate.from_messages([
            ("system", "You are a factual assistant that provides concise and accurate answers based on the provided context. "
                      "Cite the relevant parts of the context in your answer. "
                      "If the information is not in the context, say that you don't know."),
            ("human", "Context:\n{context}\n\nFactual question: {query}\n\nProvide a concise answer with citations:")
        ])
        
        # Specialized template for conceptual questions
        self.conceptual_template = ChatPromptTemplate.from_messages([
            ("system", "You are a conceptual assistant that explains complex concepts based on the provided context. "
                      "Provide clear explanations with examples when possible. "
                      "If the concept is not covered in the context, say that you don't know."),
            ("human", "Context:\n{context}\n\nConcept to explain: {query}\n\nProvide a clear explanation:")
        ])
        
        # Default template
        self.current_template = self.standard_template
    
    def process_documents(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Process documents for the RAG system.
        
        Args:
            chunk_size: Size of each chunk.
            chunk_overlap: Overlap between chunks.
        """
        logger.info("Processing documents...")
        
        # Load documents
        self.documents = self.document_processor.load_documents()
        
        # Split documents
        self.chunks = self.document_processor.split_documents(
            self.documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        
        # Generate embeddings for each embedding model
        for model_name, embedding_manager in self.embedding_managers.items():
            logger.info(f"Generating embeddings with {model_name}...")
            embeddings = embedding_manager.generate_embeddings(self.chunks)
            
            # Create vector store
            vector_store = VectorStore(embedding_manager.dimension)
            vector_store.add_documents(self.chunks, embeddings)
            self.vector_stores[model_name] = vector_store
            
            # Save vector store
            os.makedirs("vector_stores", exist_ok=True)
            vector_store.save(f"vector_stores/{model_name}")
        
        logger.info("Document processing complete")
    
    def set_embedding_model(self, model_name: str):
        """Set the current embedding model.
        
        Args:
            model_name: Name of the embedding model to use.
        """
        if model_name in self.embedding_managers:
            self.current_embedding_manager = self.embedding_managers[model_name]
            logger.info(f"Set embedding model to {model_name}")
        else:
            logger.error(f"Embedding model {model_name} not found")
    
    def set_prompt_template(self, template_type: str):
        """Set the prompt template to use.
        
        Args:
            template_type: Type of prompt template to use.
        """
        if template_type == "standard":
            self.current_template = self.standard_template
        elif template_type == "factual":
            self.current_template = self.factual_template
        elif template_type == "conceptual":
            self.current_template = self.conceptual_template
        else:
            logger.error(f"Unknown template type: {template_type}")
            return
        
        logger.info(f"Set prompt template to {template_type}")
    
    def retrieve(self, query: str, k: int = 4, 
                 retrieval_strategy: str = "similarity", 
                 metadata_filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Query to retrieve documents for.
            k: Number of documents to retrieve.
            retrieval_strategy: Strategy to use for retrieval.
            metadata_filter: Filter to apply to document metadata.
            
        Returns:
            List of retrieved documents.
        """
        logger.info(f"Retrieving documents for query: '{query}' using {retrieval_strategy} strategy")
        
        # Get current vector store
        vector_store = self.vector_stores.get(self.current_embedding_manager.model_name)
        
        if not vector_store:
            logger.error("Vector store not initialized")
            return []
        
        # Generate query embedding
        query_embedding = self.current_embedding_manager.generate_query_embedding(query)
        
        # Retrieve documents based on strategy
        if retrieval_strategy == "similarity":
            results = vector_store.similarity_search(query_embedding, k=k)
        elif retrieval_strategy == "mmr":
            results = vector_store.maximum_marginal_relevance_search(query_embedding, k=k)
        elif retrieval_strategy == "hybrid":
            results = vector_store.hybrid_search(query, self.chunks, k=k)
        else:
            logger.error(f"Unknown retrieval strategy: {retrieval_strategy}")
            return []
        
        # Apply metadata filter if provided
        if metadata_filter:
            filtered_results = []
            for doc, score in results:
                match = True
                for key, value in metadata_filter.items():
                    if key not in doc.metadata or doc.metadata[key] != value:
                        match = False
                        break
                if match:
                    filtered_results.append((doc, score))
            results = filtered_results
        
        # Log and return documents
        retrieved_docs = [doc for doc, _ in results]
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        return retrieved_docs
    
    def generate(self, query: str, context_docs: List[Document]) -> str:
        """Generate a response using retrieved context.
        
        Args:
            query: User query.
            context_docs: Retrieved context documents.
            
        Returns:
            Generated response.
        """
        logger.info("Generating response...")
        
        if not context_docs:
            logger.warning("No context documents provided")
            return "I don't have enough information to answer that question."
        
        # Format context
        context = "\n\n".join([f"Document {i+1} (Source: {doc.metadata.get('source', 'Unknown')}): {doc.page_content}" 
                              for i, doc in enumerate(context_docs)])
        
        # Generate response using the current template
        chain = self.current_template | self.llm | StrOutputParser()
        
        response = chain.invoke({"context": context, "query": query})
        
        return response
    
    def query(self, query: str, k: int = 4, 
              retrieval_strategy: str = "similarity",
              template_type: str = "standard",
              metadata_filter: Optional[Dict[str, Any]] = None) -> Tuple[str, List[Document]]:
        """Query the RAG system.
        
        Args:
            query: User query.
            k: Number of documents to retrieve.
            retrieval_strategy: Strategy to use for retrieval.
            template_type: Type of prompt template to use.
            metadata_filter: Filter to apply to document metadata.
            
        Returns:
            Generated response and retrieved documents.
        """
        # Set prompt template
        self.set_prompt_template(template_type)
        
        # Retrieve documents
        retrieved_docs = self.retrieve(query, k=k, retrieval_strategy=retrieval_strategy, metadata_filter=metadata_filter)
        
        # Generate response
        response = self.generate(query, retrieved_docs)
        
        return response, retrieved_docs
    
    def rewrite_query(self, query: str) -> str:
        """Rewrite a query to improve retrieval performance.
        
        Args:
            query: Original query.
            
        Returns:
            Rewritten query.
        """
        logger.info(f"Rewriting query: '{query}'")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a search query optimization assistant. Your task is to rewrite the given query to make it more effective for retrieval from a document database. Keep the rewritten query concise and focused on the key information needs."),
            ("human", "Original query: {query}\n\nRewrite this query to be more effective for retrieving relevant information:")
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        rewritten_query = chain.invoke({"query": query})
        
        logger.info(f"Rewritten query: '{rewritten_query}'")
        return rewritten_query
    
    def filter_retrieved_docs(self, query: str, docs: List[Document], threshold: float = 0.3) -> List[Document]:
        """Filter retrieved documents to remove irrelevant ones.
        
        Args:
            query: User query.
            docs: Retrieved documents.
            threshold: Relevance threshold.
            
        Returns:
            Filtered documents.
        """
        logger.info(f"Filtering {len(docs)} retrieved documents")
        
        if not docs:
            return []
        
        # Encode query and documents
        query_embedding = self.current_embedding_manager.generate_query_embedding(query)
        doc_embeddings = self.current_embedding_manager.model.encode([doc.page_content for doc in docs])
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        # Filter based on threshold
        filtered_docs = [doc for i, doc in enumerate(docs) if similarities[i] > threshold]
        
        logger.info(f"Filtered to {len(filtered_docs)} documents")
        return filtered_docs
    
    def hierarchical_retrieval(self, query: str, k: int = 4) -> List[Document]:
        """Perform hierarchical retrieval.
        
        Args:
            query: User query.
            k: Number of documents to retrieve.
            
        Returns:
            Retrieved documents.
        """
        logger.info(f"Performing hierarchical retrieval for query: '{query}'")
        
        # Step 1: Coarse retrieval with a larger k
        coarse_k = k * 3
        coarse_docs = self.retrieve(query, k=coarse_k, retrieval_strategy="similarity")
        
        if not coarse_docs:
            return []
        
        # Step 2: Fine-grained retrieval within the coarse results
        # Create a temporary vector store with just the coarse results
        temp_docs = coarse_docs
        temp_embeddings = self.current_embedding_manager.generate_embeddings(temp_docs)
        
        temp_vector_store = VectorStore(self.current_embedding_manager.dimension)
        temp_vector_store.add_documents(temp_docs, temp_embeddings)
        
        # Query the temporary vector store
        query_embedding = self.current_embedding_manager.generate_query_embedding(query)
        fine_results = temp_vector_store.similarity_search(query_embedding, k=k)
        
        fine_docs = [doc for doc, _ in fine_results]
        
        logger.info(f"Hierarchical retrieval returned {len(fine_docs)} documents")
        return fine_docs
    
    def self_correcting_rag(self, query: str, k: int = 4) -> str:
        """Implement self-correcting RAG.
        
        Args:
            query: User query.
            k: Number of documents to retrieve.
            
        Returns:
            Generated response.
        """
        logger.info(f"Performing self-correcting RAG for query: '{query}'")
        
        # Initial retrieval and generation
        retrieved_docs = self.retrieve(query, k=k, retrieval_strategy="mmr")
        initial_response = self.generate(query, retrieved_docs)
        
        # Evaluate the response
        evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a critical evaluator. Assess whether the given answer adequately addresses the question based on the provided context. If it doesn't, identify what's missing or incorrect."),
            ("human", "Question: {query}\n\nContext:\n{context}\n\nAnswer: {answer}\n\nIs this answer adequate, accurate, and complete based on the context? If not, what's missing or incorrect?")
        ])
        
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        evaluation_chain = evaluation_prompt | self.llm | StrOutputParser()
        
        evaluation = evaluation_chain.invoke({
            "query": query,
            "context": context,
            "answer": initial_response
        })
        
        # If evaluation suggests the answer is inadequate, regenerate
        if "inadequate" in evaluation.lower() or "incomplete" in evaluation.lower() or "incorrect" in evaluation.lower():
            logger.info("Initial response deemed inadequate, regenerating...")
            
            # Try a different retrieval strategy or more documents
            additional_docs = self.retrieve(query, k=k+2, retrieval_strategy="hybrid")
            all_docs = list(set(retrieved_docs + additional_docs))
            
            # Generate improved response
            improvement_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that provides accurate and complete answers. The previous answer to this question was deemed inadequate for these reasons: {evaluation}. Using the provided context, generate an improved answer."),
                ("human", "Context:\n{context}\n\nQuestion: {query}\n\nProvide an improved answer:")
            ])
            
            context = "\n\n".join([doc.page_content for doc in all_docs])
            improvement_chain = improvement_prompt | self.llm | StrOutputParser()
            
            improved_response = improvement_chain.invoke({
                "query": query,
                "context": context,
                "evaluation": evaluation
            })
            
            return improved_response
        
        # If evaluation is good, return the initial response
        return initial_response


class RAGEvaluator:
    """Class for evaluating RAG system performance."""
    
    def __init__(self, rag_system: RAGSystem):
        """Initialize the RAG evaluator.
        
        Args:
            rag_system: RAG system to evaluate.
        """
        self.rag_system = rag_system
    
    def evaluate_retrieval(self, query: str, relevant_doc_ids: List[str], 
                          k: int = 4, retrieval_strategy: str = "similarity") -> Dict[str, float]:
        """Evaluate retrieval performance.
        
        Args:
            query: Query to evaluate.
            relevant_doc_ids: IDs of relevant documents.
            k: Number of documents to retrieve.
            retrieval_strategy: Retrieval strategy to use.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Retrieve documents
        retrieved_docs = self.rag_system.retrieve(query, k=k, retrieval_strategy=retrieval_strategy)
        
        # Calculate precision, recall, and F1
        retrieved_ids = [doc.metadata.get("source", "") for doc in retrieved_docs]
        
        tp = len(set(retrieved_ids).intersection(set(relevant_doc_ids)))
        fp = len(retrieved_ids) - tp
        fn = len(relevant_doc_ids) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        logger.info(f"Retrieval evaluation: precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def evaluate_answer_quality(self, query: str, answer: str, ground_truth: str) -> float:
        """Evaluate answer quality using LLM as a judge.
        
        Args:
            query: Query that was answered.
            answer: Generated answer.
            ground_truth: Ground truth answer.
            
        Returns:
            Quality score from 0 to 1.
        """
        prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an objective evaluator of answer quality. Given a question, a generated answer, and a ground truth answer, rate the quality of the generated answer on a scale from 0 to 1, where:\n"
              "- 0: Completely incorrect or irrelevant\n"
              "- 0.25: Partially correct but missing key information or contains significant errors\n"
              "- 0.5: Moderately correct with some errors or omissions\n"
              "- 0.75: Mostly correct with minor errors or omissions\n"
              "- 1: Perfectly correct and complete\n\n"
              "Provide your rating as a single float value followed by a brief explanation."),
    ("human", "Question: {query}\n\nGenerated Answer: {answer}\n\nGround Truth: {ground_truth}\n\nRate the quality of the generated answer (0-1):")
])

        chain = prompt | self.rag_system.llm | StrOutputParser()

        evaluation = chain.invoke({
            "query": query,
            "answer": answer,
            "ground_truth": ground_truth
        })
        
        # Extract the numeric score from the response
        try:
            score_text = evaluation.split('\n')[0].strip()
            score = float(score_text)
            score = max(0, min(1, score))  # Clamp between 0 and 1
        except ValueError:
            # If we can't extract a score, use a default value
            logger.warning(f"Could not extract score from evaluation: {evaluation}")
            score = 0.5
        
        logger.info(f"Answer quality evaluation: score={score:.3f}")
        
        return score
    
    def create_test_queries(self, num_queries: int = 5) -> List[Dict[str, Any]]:
        """Create test queries for evaluation.
        
        Args:
            num_queries: Number of test queries to create.
            
        Returns:
            List of test query dictionaries.
        """
        # For a real system, these would be manually created or derived from document content
        test_queries = [
            {
                "query": "What is RAG?",
                "relevant_doc_ids": [],  # Would be filled with actual doc IDs
                "ground_truth": "RAG (Retrieval-Augmented Generation) is a technique that enhances LLM capabilities by retrieving relevant information from external knowledge sources before generating responses."
            },
            {
                "query": "How does document chunking work?",
                "relevant_doc_ids": [],
                "ground_truth": "Document chunking involves splitting documents into smaller, manageable pieces with overlap between chunks to prevent information loss at boundaries."
            },
            {
                "query": "What is Maximum Marginal Relevance?",
                "relevant_doc_ids": [],
                "ground_truth": "Maximum Marginal Relevance (MMR) is a retrieval strategy that balances relevance and diversity in search results by selecting documents that are both relevant to the query and different from already selected documents."
            },
            {
                "query": "What is the difference between semantic search and keyword search?",
                "relevant_doc_ids": [],
                "ground_truth": "Semantic search uses vector embeddings to find documents with similar meaning regardless of exact wording, while keyword search finds documents containing specific words from the query."
            },
            {
                "query": "What are the components of a RAG system?",
                "relevant_doc_ids": [],
                "ground_truth": "A RAG system typically includes document processing (loading and chunking), embedding generation, vector storage, retrieval strategies, and integration with LLMs for generating responses."
            }
        ]
        
        return test_queries[:num_queries]
    
    def evaluate_configuration(self, 
                             embedding_model: str,
                             retrieval_strategy: str,
                             template_type: str,
                             test_queries: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate a specific RAG configuration.
        
        Args:
            embedding_model: Embedding model to use.
            retrieval_strategy: Retrieval strategy to use.
            template_type: Prompt template type to use.
            test_queries: List of test queries to evaluate on.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info(f"Evaluating configuration: embedding={embedding_model}, retrieval={retrieval_strategy}, template={template_type}")
        
        # Set configuration
        self.rag_system.set_embedding_model(embedding_model)
        self.rag_system.set_prompt_template(template_type)
        
        # Run evaluation on test queries
        precision_sum = 0
        recall_sum = 0
        f1_sum = 0
        quality_sum = 0
        
        for test_query in test_queries:
            query = test_query["query"]
            relevant_doc_ids = test_query["relevant_doc_ids"]
            ground_truth = test_query["ground_truth"]
            
            # Evaluate retrieval
            if relevant_doc_ids:
                retrieval_metrics = self.evaluate_retrieval(
                    query, relevant_doc_ids, retrieval_strategy=retrieval_strategy
                )
                precision_sum += retrieval_metrics["precision"]
                recall_sum += retrieval_metrics["recall"]
                f1_sum += retrieval_metrics["f1"]
            
            # Generate answer
            answer, _ = self.rag_system.query(
                query, retrieval_strategy=retrieval_strategy, template_type=template_type
            )
            
            # Evaluate answer quality
            quality = self.evaluate_answer_quality(query, answer, ground_truth)
            quality_sum += quality
        
        # Calculate averages
        num_queries = len(test_queries)
        precision_avg = precision_sum / num_queries if relevant_doc_ids else None
        recall_avg = recall_sum / num_queries if relevant_doc_ids else None
        f1_avg = f1_sum / num_queries if relevant_doc_ids else None
        quality_avg = quality_sum / num_queries
        
        # Log results
        logger.info(f"Evaluation results: precision={precision_avg:.3f if precision_avg is not None else 'N/A'}, "
                   f"recall={recall_avg:.3f if recall_avg is not None else 'N/A'}, "
                   f"f1={f1_avg:.3f if f1_avg is not None else 'N/A'}, "
                   f"quality={quality_avg:.3f}")
        
        return {
            "precision": precision_avg,
            "recall": recall_avg,
            "f1": f1_avg,
            "quality": quality_avg
        }
    
    def compare_configurations(self, test_queries: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Compare different RAG configurations.
        
        Args:
            test_queries: List of test queries to evaluate on.
            
        Returns:
            Dictionary of configuration names to evaluation metrics.
        """
        logger.info("Comparing different RAG configurations")
        
        # Define configurations to test
        configurations = [
            {
                "name": "MiniLM-similarity-standard",
                "embedding_model": "all-MiniLM-L6-v2",
                "retrieval_strategy": "similarity",
                "template_type": "standard"
            },
            {
                "name": "MiniLM-mmr-standard",
                "embedding_model": "all-MiniLM-L6-v2",
                "retrieval_strategy": "mmr",
                "template_type": "standard"
            },
            {
                "name": "MPNet-similarity-standard",
                "embedding_model": "all-mpnet-base-v2",
                "retrieval_strategy": "similarity",
                "template_type": "standard"
            },
            {
                "name": "MiniLM-similarity-factual",
                "embedding_model": "all-MiniLM-L6-v2",
                "retrieval_strategy": "similarity",
                "template_type": "factual"
            },
            {
                "name": "MiniLM-hybrid-standard",
                "embedding_model": "all-MiniLM-L6-v2",
                "retrieval_strategy": "hybrid",
                "template_type": "standard"
            }
        ]
        
        # Evaluate each configuration
        results = {}
        
        for config in configurations:
            metrics = self.evaluate_configuration(
                config["embedding_model"],
                config["retrieval_strategy"],
                config["template_type"],
                test_queries
            )
            
            results[config["name"]] = metrics
        
        # Log comparison results
        logger.info("Configuration comparison results:")
        for config_name, metrics in results.items():
            quality = metrics.get("quality", 0)
            f1 = metrics.get("f1", 0) or 0  # Handle None
            logger.info(f"{config_name}: quality={quality:.3f}, f1={f1:.3f}")
        
        return results
    
    def plot_comparison_results(self, results: Dict[str, Dict[str, float]], output_file: str = "comparison_results.png"):
        """Plot comparison results.
        
        Args:
            results: Results from compare_configurations.
            output_file: Path to save the plot to.
        """
        # Extract configuration names and metrics
        config_names = list(results.keys())
        quality_scores = [results[name].get("quality", 0) for name in config_names]
        f1_scores = [results[name].get("f1", 0) or 0 for name in config_names]  # Handle None
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(config_names))
        width = 0.35
        
        plt.bar(x - width/2, quality_scores, width, label='Answer Quality')
        plt.bar(x + width/2, f1_scores, width, label='Retrieval F1')
        
        plt.xlabel('Configuration')
        plt.ylabel('Score')
        plt.title('RAG Configuration Comparison')
        plt.xticks(x, config_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_file)
        logger.info(f"Saved comparison results plot to {output_file}")


def main():
    """Main function to run the RAG system."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create RAG system
    rag_system = RAGSystem()
    
    # Process documents
    rag_system.process_documents(chunk_size=1000, chunk_overlap=200)
    
    # Set up evaluator
    evaluator = RAGEvaluator(rag_system)
    
    # Create test queries
    test_queries = evaluator.create_test_queries()
    
    # Compare configurations
    results = evaluator.compare_configurations(test_queries)
    
    # Plot results
    evaluator.plot_comparison_results(results)
    
    # Run interactive query loop
    print("\n=== RAG System Query Interface ===")
    print("Enter your query (or 'exit' to quit):")
    
    while True:
        query = input("> ")
        
        if query.lower() in ["exit", "quit", "q"]:
            break
        
        # Process query
        response, retrieved_docs = rag_system.query(
            query,
            k=4, 
            retrieval_strategy="mmr",
            template_type="standard"
        )
        
        # Print response
        print("\nRAG Response:")
        print(response)
        
        # Print retrieved documents
        print("\nRetrieved Documents:")
        for i, doc in enumerate(retrieved_docs[:3]):  # Show top 3 docs
            source = doc.metadata.get("source", "Unknown")
            print(f"\n[Document {i+1}] Source: {source}")
            content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            print(content_preview)
        
        print("\n" + "-" * 50)


if __name__ == "__main__":
    main()