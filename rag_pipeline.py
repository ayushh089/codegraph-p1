from chunking import GraphChunker
from embeddings import EmbeddingStore
from neo4j import GraphDatabase
import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class RAGPipeline:
    """Complete RAG pipeline: Graph + Vector + LLM (Gemini via OpenAI compatible API)"""
    
    def __init__(self):
        print("🚀 Initializing RAG Pipeline...")
        self.chunker = GraphChunker()
        self.vector_store = EmbeddingStore(fresh_start=False)
        
        # Initialize Gemini via OpenAI-compatible endpoint
        self.llm = OpenAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.model = "gemini-2.0-flash"
        
        self.neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )
        
        print("✅ Gemini API ready! Using model: " + self.model)
    
    def index_graph(self):
        """Step 1: Extract all data from Neo4j and create embeddings"""
        print("\n" + "="*50)
        print("STEP 1: Indexing Graph Data")
        print("="*50)
        
        try:
            chunks = self.chunker.get_all_chunks()
            
            if not chunks:
                print("❌ No chunks found! Make sure Neo4j has data.")
                print("   Run: python main.py test_repo first")
                return False
            
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
        
        context = {
            'vector_matches': vector_matches,
            'graph_matches': graph_matches
        }
        
        return context
    
    def _graph_search(self, query):
        """Search Neo4j for exact matches"""
        keywords = query.lower().split()
        
        with self.neo4j_driver.session() as session:
            result = session.run("""
                MATCH (f:Function)
                WHERE any(keyword IN $keywords WHERE toLower(f.name) CONTAINS keyword)
                RETURN f.name as name, f.file as file
                LIMIT 5
            """, keywords=keywords)
            
            matches = [dict(record) for record in result]
        
        return matches
    
    def _get_filename(self, filepath):
        """Extract filename from full path safely"""
        return os.path.basename(filepath)
    
    def _answer_from_neo4j_direct(self, question):
        """Direct Neo4j queries for common question types - NO LLM"""
        question_lower = question.lower()
        
        # Count classes
        if 'class' in question_lower and ('how many' in question_lower or 'count' in question_lower):
            with self.neo4j_driver.session() as session:
                result = session.run("MATCH (c:Class) RETURN count(c) as count")
                count = result.single()['count']
                if count > 0:
                    classes = session.run("MATCH (c:Class) RETURN c.name as name, c.file as file")
                    class_list = []
                    for record in classes:
                        filename = self._get_filename(record['file'])
                        class_list.append(f"{record['name']} in {filename}")
                    return f"Found {count} class(es): {', '.join(class_list)}"
                return "Found 0 classes in the codebase."
        
        # Count functions
        if 'function' in question_lower and ('how many' in question_lower or 'count' in question_lower):
            with self.neo4j_driver.session() as session:
                result = session.run("MATCH (f:Function) RETURN count(f) as count")
                count = result.single()['count']
                return f"Found {count} function(s) in the codebase."
        
        # Which file calls X?
        if 'calls' in question_lower:
            match = re.search(r'calls (\w+)', question_lower)
            if match:
                func_name = match.group(1)
                with self.neo4j_driver.session() as session:
                    result = session.run("""
                        MATCH (caller:Function)-[:CALLS]->(callee:Function {name: $name})
                        RETURN caller.name as caller, caller.file as file
                    """, name=func_name)
                    callers = list(result)
                    if callers:
                        caller_names = []
                        for c in callers:
                            filename = self._get_filename(c['file'])
                            caller_names.append(f"{c['caller']} (in {filename})")
                        return f"Function '{func_name}' is called by: {', '.join(caller_names)}"
                    return f"No functions call '{func_name}'"
        
        # List all files
        if 'file' in question_lower and ('list' in question_lower or 'all' in question_lower):
            with self.neo4j_driver.session() as session:
                result = session.run("MATCH (f:File) RETURN f.path as file")
                files = [self._get_filename(record['file']) for record in result]
                return f"Files in codebase: {', '.join(files)}"
        
        return None
    
    def generate_answer(self, query, context):
        """Step 3: Generate answer using Gemini"""
        
        # First try direct Neo4j query (NO LLM)
        direct_answer = self._answer_from_neo4j_direct(query)
        if direct_answer:
            return direct_answer
        
        # Otherwise use Gemini
        formatted_context = self._format_context_for_llm(context)
        
        prompt = f"""You are a code analysis system. Answer based ONLY on the context below.

CONTEXT:
{formatted_context}

QUESTION: {query}

RULES:
1. ONLY use information from CONTEXT
2. If answer not in context, say "Not found in codebase"
3. Be specific - mention function names and file locations

ANSWER:"""
        
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a codebase expert. Answer only from given context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️  Gemini error: {e}")
            return self._format_search_results(context)
    
    def _format_context_for_llm(self, context):
        """Format retrieved context for LLM"""
        text = ""
        
        if context['graph_matches']:
            text += "EXACT MATCHES:\n"
            for match in context['graph_matches']:
                filename = self._get_filename(match['file'])
                text += f"• {match['name']} -> in: {filename}\n"
            text += "\n"
        
        if context['vector_matches']:
            text += "CODE CHUNKS:\n"
            for i, doc in enumerate(context['vector_matches'][:3]):
                doc_preview = doc[:400] + "..." if len(doc) > 400 else doc
                text += f"\n[{i+1}]\n{doc_preview}\n"
        
        if not text:
            text = "No relevant code found."
        
        return text
    
    def _format_search_results(self, context):
        """Format search results as plain text (fallback)"""
        result_text = "🔍 SEARCH RESULTS:\n\n"
        
        if context['graph_matches']:
            result_text += "Exact matches:\n"
            for match in context['graph_matches']:
                filename = self._get_filename(match['file'])
                result_text += f"  • {match['name']} - in {filename}\n"
        
        if context['vector_matches']:
            result_text += f"\nFound {len(context['vector_matches'])} similar chunks.\n"
        
        return result_text
    
    def ask(self, question):
        """Main method: Ask a question and get answer"""
        print("\n" + "="*60)
        print(f"🤔 QUESTION: {question}")
        print("="*60)
        
        context = self.retrieve_context(question)
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
if __name__ == "__main__":
    import sys
    
    reindex = False
    if len(sys.argv) > 1 and sys.argv[1] == "--reindex":
        reindex = True
        print("🔄 --reindex flag detected. Will re-index all data.")
    
    rag = RAGPipeline()
    
    if reindex:
        print("\n🗑️  Clearing existing embeddings...")
        rag.vector_store.clear_all()
        rag.vector_store = EmbeddingStore(fresh_start=True)
    
    if rag.vector_store.collection.count() == 0:
        print("\n📚 No existing embeddings found. Indexing graph data...")
        success = rag.index_graph()
        if not success:
            print("\n❌ Indexing failed. Please run 'python main.py test_repo' first.")
            exit(1)
    else:
        print(f"\n✅ Found {rag.vector_store.collection.count()} existing chunks in database")
    
    print("\n✅ Ready for questions! Using Gemini via OpenAI compatible API")
    
    while True:
        print("\n" + "-"*60)
        question = input("\n❓ Ask about your codebase (or 'quit' to exit): ")
        
        if question.lower() == "quit":
            break
        
        if question.strip():
            rag.ask(question)
    
    rag.close()