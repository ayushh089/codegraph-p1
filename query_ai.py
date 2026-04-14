#!/usr/bin/env python3
from rag_pipeline import RAGPipeline
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python query_ai.py 'your question here'")
        print("\nExamples:")
        print("  python query_ai.py 'What functions handle user login?'")
        print("  python query_ai.py 'Show me all database operations'")
        print("  python query_ai.py 'Which functions call calculate_total?'")
        return
    
    query = sys.argv[1]
    
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Check if we need to index first
    if rag.vector_store.collection.count() == 0:
        print("📚 First run: Indexing graph data...")
        rag.index_graph()
    
    # Ask question
    answer = rag.ask(query)
    
    rag.close()

if __name__ == "__main__":
    main()