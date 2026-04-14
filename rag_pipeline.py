from chunking import GraphChunker
from embeddings import EmbeddingStore
from neo4j import GraphDatabase
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    """Complete RAG pipeline: Graph + Vector + LLM (Ollama)"""
    
    def __init__(self):
        print("🚀 Initializing RAG Pipeline...")
        self.chunker = GraphChunker()
        self.vector_store = EmbeddingStore(fresh_start=False)  # Don't auto-delete
        
        # Ollama configuration
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "llama3.2:3b"
        self.llm_available = self._check_ollama()
        
        self.neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )
    
    def _check_ollama(self):
        """Check if Ollama is running"""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.model not in str(model_names):
                    print(f"⚠️  Model {self.model} not found. Pulling it now...")
                    os.system(f"ollama pull {self.model}")
                
                print(f"✅ Ollama is ready! Using model: {self.model}")
                return True
            return False
        except:
            print("❌ Ollama is not running!")
            print("   Please start Ollama from Start Menu or system tray")
            return False
    
    def _call_ollama(self, prompt):
        """Call Ollama API with longer timeout"""
        if not self.llm_available:
            return None
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,
                    "max_tokens": 300
                },
                timeout=120  # Increase from 30 to 120 seconds
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                print(f"⚠️  Ollama error: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"⚠️  Ollama timeout - model is still thinking...")
            print(f"   Try again or use a smaller model")
            return None
        except Exception as e:
            print(f"⚠️  Ollama error: {e}")
            return None
    
    def index_graph(self):
        """Step 1: Extract all data from Neo4j and create embeddings"""
        print("\n" + "="*50)
        print("STEP 1: Indexing Graph Data")
        print("="*50)
        
        try:
            # Get chunks from graph
            chunks = self.chunker.get_all_chunks()
            
            if not chunks:
                print("❌ No chunks found! Make sure Neo4j has data.")
                print("   Run: python main.py test_repo first")
                return False
            
            # Create embeddings and store
            self.vector_store.create_embeddings(chunks)
            
            print("\n✅ Indexing complete!")
            return True
            
        except Exception as e:
            print(f"❌ Error during indexing: {e}")
            return False
    
    def retrieve_context(self, query, top_k=5):
        """Step 2: Retrieve relevant context using hybrid search"""
        
        print(f"\n🔍 Searching for: '{query}'")
        
        # Method 1: Vector search (semantic)
        try:
            vector_results = self.vector_store.search_similar(query, top_k)
            vector_matches = vector_results['documents'][0] if vector_results['documents'] else []
        except Exception as e:
            print(f"⚠️  Vector search error: {e}")
            vector_matches = []
        
        # Method 2: Graph search (exact relationships)
        try:
            graph_matches = self._graph_search(query)
        except Exception as e:
            print(f"⚠️  Graph search error: {e}")
            graph_matches = []
        
        # Combine results
        context = {
            'vector_matches': vector_matches,
            'graph_matches': graph_matches
        }
        
        return context
    
    def _graph_search(self, query):
        """Search Neo4j for exact matches"""
        
        # Extract keywords from query
        keywords = query.lower().split()
        
        with self.neo4j_driver.session() as session:
            # Search for functions with matching names
            result = session.run("""
                MATCH (f:Function)
                WHERE any(keyword IN $keywords WHERE toLower(f.name) CONTAINS keyword)
                RETURN f.name as name, f.file as file
                LIMIT 5
            """, keywords=keywords)
            
            matches = [dict(record) for record in result]
        
        return matches
    
    def generate_answer(self, query, context):
        """Step 3: Generate answer using Ollama"""
        
        # Format context for LLM
        formatted_context = self._format_context_for_llm(context)
        
        prompt = f"""You are a codebase expert. Answer the question based ONLY on the code structure below.

QUESTION: {query}

CODE CONTEXT:
{formatted_context}

INSTRUCTIONS:
1. Only use information from the context above
2. Be specific - mention function names and file locations
3. If the answer isn't in context, say "I couldn't find this in the codebase"
4. Keep answer concise but informative (max 150 words)

ANSWER:"""
        
        # Try Ollama first
        if self.llm_available:
            answer = self._call_ollama(prompt)
            if answer:
                return answer
        
        # Fallback to formatted search results
        return self._format_search_results(context)
    
    def _format_context_for_llm(self, context):
        """Format retrieved context for LLM"""
        text = ""
        
        # Add vector search results
        if context['vector_matches']:
            text += "Similar code chunks found:\n\n"
            for i, doc in enumerate(context['vector_matches'][:3]):
                # Truncate long docs
                doc_preview = doc[:500] + "..." if len(doc) > 500 else doc
                text += f"--- Chunk {i+1} ---\n{doc_preview}\n\n"
        
        # Add graph search results
        if context['graph_matches']:
            text += "\nExact matches from graph:\n"
            for match in context['graph_matches'][:3]:
                text += f"- {match['name']} (in {match['file']})\n"
        
        return text if text else "No relevant code found in the codebase."
    
    def _format_search_results(self, context):
        """Format search results as plain text (fallback)"""
        result_text = "🔍 SEARCH RESULTS:\n\n"
        
        if context['graph_matches']:
            result_text += "📌 Exact matches found:\n"
            for match in context['graph_matches']:
                result_text += f"  • {match['name']} - in {match['file']}\n"
        else:
            result_text += "📌 No exact matches found.\n"
        
        if context['vector_matches']:
            result_text += f"\n📄 Found {len(context['vector_matches'])} semantically similar code chunks:\n"
            for i, doc in enumerate(context['vector_matches'][:2]):
                result_text += f"\n--- Chunk {i+1} ---\n{doc[:300]}\n"
        
        return result_text
    
    def ask(self, question):
        """Main method: Ask a question and get answer"""
        print("\n" + "="*60)
        print(f"🤔 QUESTION: {question}")
        print("="*60)
        
        # Step 2: Retrieve context
        context = self.retrieve_context(question)
        
        # Step 3: Generate answer
        answer = self.generate_answer(question, context)
        
        print("\n💡 ANSWER:")
        print("="*60)
        print(answer)
        print("="*60)
        
        return answer
    
    def close(self):
        """Clean up connections"""
        try:
            self.chunker.close()
            self.neo4j_driver.close()
        except:
            pass

# Main execution
# Main execution
if __name__ == "__main__":
    import sys
    
    # Check for --reindex flag
    reindex = False
    if len(sys.argv) > 1 and sys.argv[1] == "--reindex":
        reindex = True
        print("🔄 --reindex flag detected. Will re-index all data.")
    
    rag = RAGPipeline()
    
    # Force re-index if requested
    if reindex:
        print("\n🗑️  Clearing existing embeddings...")
        rag.vector_store.clear_all()
        rag.vector_store = EmbeddingStore(fresh_start=True)
        rag.vector_store.collection = rag.vector_store.chroma_client.get_collection("code_chunks")
    
    # ONLY index if no chunks exist OR reindex flag is set
    if rag.vector_store.collection.count() == 0:
        print("\n📚 No existing embeddings found. Indexing graph data...")
        success = rag.index_graph()
        if not success:
            print("\n❌ Indexing failed. Please run 'python main.py test_repo' first.")
            exit(1)
    else:
        print(f"\n✅ Found {rag.vector_store.collection.count()} existing chunks in database")
        print("   Skipping indexing. Use --reindex flag to force re-index.")
    
    print("\n✅ Ready for questions! Using Ollama + " + rag.model)
    print("💡 Tip: First query might be slow as model loads into memory")
    
    # Interactive question loop
    while True:
        print("\n" + "-"*60)
        question = input("\n❓ Ask about your codebase (or 'quit' to exit): ")
        
        if question.lower() == "quit":
            break
        
        if question.strip():
            rag.ask(question)
    
    rag.close()