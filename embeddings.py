from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
from chromadb.utils import embedding_functions
import os
import hashlib
import logging
from dotenv import load_dotenv

load_dotenv()
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

class EmbeddingStore:
    """Handle embeddings using Google Gemini and vector storage"""
    
    def __init__(self, fresh_start=False):
        print("🔄 Loading Google Gemini Embedding model...")
        
        # Initialize Google Gemini Embeddings
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Create ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # ONLY delete if fresh_start = True
        if fresh_start:
            try:
                existing_collections = self.chroma_client.list_collections()
                collection_names = [col.name for col in existing_collections]
                
                if 'code_chunks' in collection_names:
                    print("🗑️  Removing existing collection for fresh start...")
                    self.chroma_client.delete_collection("code_chunks")
            except Exception as e:
                print(f"⚠️  Note: {e}")
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection("code_chunks")
            print(f"✅ Found existing collection with {self.collection.count()} chunks")
        except:
            self.collection = self.chroma_client.create_collection(
                name="code_chunks",
                embedding_function=embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                    api_key=os.getenv("GOOGLE_API_KEY")
                )
            )
            print("✅ Created new collection")
        
        print("✅ Google Gemini Embedding model loaded")
        print(f"📀 ChromaDB ready with {self.collection.count()} existing chunks")
    
    def needs_indexing(self):
        """Check if we need to index the graph"""
        return self.collection.count() == 0
    
    def create_embeddings(self, chunks):
        """Convert chunks to embeddings and store in ChromaDB"""
        
        if self.collection.count() > 0:
            print(f"\n✅ Already have {self.collection.count()} chunks indexed. Skipping...")
            return True
        
        print(f"\n🔄 Creating embeddings for {len(chunks)} chunks...")
        
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            chunk_id = hashlib.md5(chunk['id'].encode()).hexdigest()
            ids.append(chunk_id)
            documents.append(chunk['text'])
            safe_metadata = {}
            for key, value in chunk['metadata'].items():
                if value is None:
                    safe_metadata[key] = 'none'
                elif isinstance(value, list):
                    safe_metadata[key] = ', '.join(value) if value else 'none'
                else:
                    safe_metadata[key] = str(value)
            metadatas.append(safe_metadata)
        
        batch_size = 50
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas
            )
            print(f"   ✅ Processed batch {i//batch_size + 1}/{(len(ids)-1)//batch_size + 1}")
        
        print(f"✅ Stored {len(chunks)} embeddings in ChromaDB")
        print(f"📊 Total chunks in DB: {self.collection.count()}")
        
        return True
    
    def search_similar(self, query, top_k=5):
        """Search for chunks similar to query"""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return results
    
    def clear_all(self):
        """Clear all embeddings"""
        try:
            self.chroma_client.delete_collection("code_chunks")
            print("🗑️  Deleted existing collection")
        except:
            pass
        
        self.collection = self.chroma_client.create_collection(
            name="code_chunks",
            embedding_function=embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                api_key=os.getenv("GOOGLE_API_KEY")
            )
        )
        print("✅ Created fresh collection")