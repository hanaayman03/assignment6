"""
CSAI 422: Laboratory Assignment 6
RAG System Usage Examples
"""

from rag_system_implementation import RAGSystem, RAGEvaluator, logger
import logging
import os

def example_usage():
    """Example usage of the RAG system."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Starting RAG system example usage")
    
    # 1. Create RAG system
    rag_system = RAGSystem(model_name="llama3-70b-8192")  # Using Groq's Llama 3 model
    
    # 2. Process documents (assuming documents are in the './documents' folder)
    print("\n=== Processing Documents ===")
    rag_system.process_documents(chunk_size=1000, chunk_overlap=200)
    
    # 3. Example queries with different configurations
    print("\n=== Example Queries ===")
    
    # Example 1: Basic similarity search with standard prompt
    print("\nExample 1: Basic similarity search with standard prompt")
    query = "What is RAG and how does it work?"
    response, docs = rag_system.query(
        query=query,
        k=4,
        retrieval_strategy="similarity",
        template_type="standard"
    )
    print(f"Query: {query}")
    print(f"Response: {response}")
    print(f"Retrieved {len(docs)} documents")
    
    # Example 2: MMR search with conceptual prompt
    print("\nExample 2: MMR search with conceptual prompt")
    query = "Explain the difference between embedding models"
    response, docs = rag_system.query(
        query=query,
        k=4,
        retrieval_strategy="mmr",
        template_type="conceptual"
    )
    print(f"Query: {query}")
    print(f"Response: {response}")
    print(f"Retrieved {len(docs)} documents")
    
    # Example 3: Hybrid search with factual prompt
    print("\nExample 3: Hybrid search with factual prompt")
    query = "How are documents split into chunks?"
    response, docs = rag_system.query(
        query=query,
        k=4,
        retrieval_strategy="hybrid",
        template_type="factual"
    )
    print(f"Query: {query}")
    print(f"Response: {response}")
    print(f"Retrieved {len(docs)} documents")
    
    # 4. Advanced RAG techniques
    print("\n=== Advanced RAG Techniques ===")
    
    # Example 4: Query rewriting
    print("\nExample 4: Query rewriting")
    original_query = "Tell me about vector stores"
    rewritten_query = rag_system.rewrite_query(original_query)
    response, docs = rag_system.query(rewritten_query)
    print(f"Original query: {original_query}")
    print(f"Rewritten query: {rewritten_query}")
    print(f"Response: {response}")
    
    # Example 5: Hierarchical retrieval
    print("\nExample 5: Hierarchical retrieval")
    query = "What are the evaluation metrics for RAG systems?"
    hier_docs = rag_system.hierarchical_retrieval(query, k=4)
    response = rag_system.generate(query, hier_docs)
    print(f"Query: {query}")
    print(f"Response: {response}")
    print(f"Retrieved {len(hier_docs)} documents")
    
    # Example 6: Self-correcting RAG
    print("\nExample 6: Self-correcting RAG")
    query = "What are the advantages of using FAISS for vector storage?"
    response = rag_system.self_correcting_rag(query, k=4)
    print(f"Query: {query}")
    print(f"Response: {response}")
    
    # 5. Evaluation
    print("\n=== Evaluation ===")
    evaluator = RAGEvaluator(rag_system)
    
    # Create test queries
    test_queries = evaluator.create_test_queries(num_queries=3)
    
    # Compare configurations
    print("Comparing different RAG configurations...")
    results = evaluator.compare_configurations(test_queries)
    
    # Plot results
    evaluator.plot_comparison_results(results)
    print("Evaluation complete. Check the output file for comparison results.")


def interactive_mode():
    """Run the RAG system in interactive mode."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create RAG system
    rag_system = RAGSystem()
    
    # Process documents
    print("\n=== Processing Documents ===")
    rag_system.process_documents(chunk_size=1000, chunk_overlap=200)
    
    # Run interactive query loop
    print("\n=== RAG System Query Interface ===")
    print("Available configurations:")
    print("  - Embedding models: all-MiniLM-L6-v2, all-mpnet-base-v2")
    print("  - Retrieval strategies: similarity, mmr, hybrid")
    print("  - Template types: standard, factual, conceptual")
    
    # Set default configuration
    embedding_model = "all-MiniLM-L6-v2"
    retrieval_strategy = "mmr"
    template_type = "standard"
    k = 4
    
    print(f"\nDefault configuration: {embedding_model}, {retrieval_strategy}, {template_type}, k={k}")
    print("Enter your query (or 'exit' to quit, 'config' to change configuration):")
    
    while True:
        query = input("> ")
        
        if query.lower() in ["exit", "quit", "q"]:
            break
            
        elif query.lower() == "config":
            print("\n=== Change Configuration ===")
            
            # Embedding model
            print(f"Embedding model (current: {embedding_model}):")
            print("1. all-MiniLM-L6-v2")
            print("2. all-mpnet-base-v2")
            model_choice = input("Choose model (1-2, or press Enter to keep current): ")
            if model_choice == "1":
                embedding_model = "all-MiniLM-L6-v2"
            elif model_choice == "2":
                embedding_model = "all-mpnet-base-v2"
            
            # Retrieval strategy
            print(f"\nRetrieval strategy (current: {retrieval_strategy}):")
            print("1. similarity")
            print("2. mmr")
            print("3. hybrid")
            strategy_choice = input("Choose strategy (1-3, or press Enter to keep current): ")
            if strategy_choice == "1":
                retrieval_strategy = "similarity"
            elif strategy_choice == "2":
                retrieval_strategy = "mmr"
            elif strategy_choice == "3":
                retrieval_strategy = "hybrid"
            
            # Template type
            print(f"\nTemplate type (current: {template_type}):")
            print("1. standard")
            print("2. factual")
            print("3. conceptual")
            template_choice = input("Choose template (1-3, or press Enter to keep current): ")
            if template_choice == "1":
                template_type = "standard"
            elif template_choice == "2":
                template_type = "factual"
            elif template_choice == "3":
                template_type = "conceptual"
            
            # Number of documents
            k_choice = input(f"\nNumber of documents to retrieve (current: {k}): ")
            if k_choice.isdigit() and int(k_choice) > 0:
                k = int(k_choice)
            
            # Set configuration
            rag_system.set_embedding_model(embedding_model)
            print(f"\nConfiguration updated: {embedding_model}, {retrieval_strategy}, {template_type}, k={k}")
            
        else:
            # Process query
            try:
                print("\nProcessing query...")
                response, retrieved_docs = rag_system.query(
                    query,
                    k=k, 
                    retrieval_strategy=retrieval_strategy,
                    template_type=template_type
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
            except Exception as e:
                print(f"Error processing query: {e}")
            
            print("\n" + "-" * 50)


if __name__ == "__main__":
    print("RAG System Implementation")
    print("1. Run example usage")
    print("2. Run interactive mode")
    choice = input("Enter your choice (1-2): ")
    
    if choice == "1":
        example_usage()
    elif choice == "2":
        interactive_mode()
    else:
        print("Invalid choice. Exiting.")