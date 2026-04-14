from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

class GraphChunker:
    """Extract data from Neo4j and convert to text chunks"""
    
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )
    
    def close(self):
        self.driver.close()
    
    def get_all_files(self):
        """Fetch all files from Neo4j"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (f:File)
                OPTIONAL MATCH (f)-[:CONTAINS]->(func:Function)
                OPTIONAL MATCH (f)-[:CONTAINS]->(c:Class)
                RETURN DISTINCT 
                    f.path as file_path,
                    COLLECT(DISTINCT func.name) as functions,
                    COLLECT(DISTINCT c.name) as classes
            """)
            
            files = []
            for record in result:
                # Create a text chunk for each file
                chunk = self._create_file_chunk(record)
                files.append({
                    'id': record['file_path'],
                    'text': chunk,
                    'metadata': {
                        'type': 'file',
                        'path': record['file_path'],
                        'functions': ', '.join(record['functions']) if record['functions'] else 'none',
                        'classes': ', '.join(record['classes']) if record['classes'] else 'none'
                    }
                })
            
            return files
    
    def get_all_functions(self):
        """Fetch all functions with their relationships"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (f:Function)
                OPTIONAL MATCH (f)-[:CALLS]->(called:Function)
                OPTIONAL MATCH (caller:Function)-[:CALLS]->(f)
                OPTIONAL MATCH (c:Class)-[:CONTAINS]->(f)
                OPTIONAL MATCH (file:File)-[:CONTAINS]->(f)
                
                RETURN DISTINCT 
                    f.name as function_name,
                    f.file as file_path,
                    COLLECT(DISTINCT called.name) as calls,
                    COLLECT(DISTINCT caller.name) as called_by,
                    COLLECT(DISTINCT c.name) as belongs_to_class,
                    COLLECT(DISTINCT file.path) as file_location
            """)
            
            functions = []
            for record in result:
                # Convert empty lists to ['none'] for ChromaDB compatibility
                calls = record['calls'] if record['calls'] else ['none']
                called_by = record['called_by'] if record['called_by'] else ['none']
                belongs_to_class = record['belongs_to_class'] if record['belongs_to_class'] else ['none']
                file_location = record['file_location'] if record['file_location'] else ['unknown']
                
                # Create a text chunk for each function
                chunk = self._create_function_chunk(record)
                functions.append({
                    'id': record['function_name'],
                    'text': chunk,
                    'metadata': {
                        'type': 'function',
                        'name': record['function_name'],
                        'file': record['file_path'] or 'unknown',
                        'calls': ', '.join(calls),
                        'called_by': ', '.join(called_by),
                        'belongs_to_class': ', '.join(belongs_to_class),
                        'file_location': ', '.join(file_location)
                    }
                })
            
            return functions
    
    def get_all_classes(self):
        """Fetch all classes with their methods"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Class)
                OPTIONAL MATCH (c)-[:CONTAINS]->(m:Function)
                OPTIONAL MATCH (file:File)-[:CONTAINS]->(c)
                
                RETURN DISTINCT 
                    c.name as class_name,
                    c.file as file_path,
                    COLLECT(DISTINCT m.name) as methods,
                    COLLECT(DISTINCT file.path) as file_location
            """)
            
            classes = []
            for record in result:
                # Convert empty lists to ['none']
                methods = record['methods'] if record['methods'] else ['none']
                file_location = record['file_location'] if record['file_location'] else ['unknown']
                
                chunk = self._create_class_chunk(record)
                classes.append({
                    'id': record['class_name'],
                    'text': chunk,
                    'metadata': {
                        'type': 'class',
                        'name': record['class_name'],
                        'file': record['file_path'] or 'unknown',
                        'methods': ', '.join(methods),
                        'file_location': ', '.join(file_location)
                    }
                })
            
            return classes
    
    def _create_file_chunk(self, record):
        """Convert file data to readable text"""
        functions_text = ', '.join(record['functions']) if record['functions'] else 'no functions'
        classes_text = ', '.join(record['classes']) if record['classes'] else 'no classes'
        
        chunk = f"""
FILE: {record['file_path']}

This file contains the following functions: {functions_text}

This file contains the following classes: {classes_text}
"""
        return chunk.strip()
    
    def _create_function_chunk(self, record):
        """Convert function data to readable text"""
        calls_text = ', '.join(record['calls']) if record['calls'] else 'nothing'
        called_by_text = ', '.join(record['called_by']) if record['called_by'] else 'nothing'
        
        # Get the file path for the caller
        chunk = f"""
    FUNCTION: {record['function_name']}
    DEFINED IN FILE: {record['file_path']}

    CALLERS (functions that call THIS function):
    {called_by_text}

    CALLEES (functions THIS function calls):
    {calls_text}

    IMPORTANT: If a caller is listed above, it means that caller function is in a FILE. 
    For example, if caller is 'chat', look at the caller's file location in the graph.
    """
        return chunk.strip()
    
    def _create_class_chunk(self, record):
        """Convert class data to readable text"""
        methods_text = ', '.join(record['methods']) if record['methods'] else 'No methods defined'
        
        chunk = f"""
CLASS: {record['class_name']}
Location: {record['file_path']}

Methods: {methods_text}
"""
        return chunk.strip()
    
    def get_all_chunks(self):
        """Get all chunks from graph (files + functions + classes)"""
        print("📚 Extracting chunks from Neo4j...")
        
        files = self.get_all_files()
        functions = self.get_all_functions()
        classes = self.get_all_classes()
        
        all_chunks = files + functions + classes
        
        print(f"✅ Created {len(all_chunks)} chunks")
        print(f"   - {len(files)} file chunks")
        print(f"   - {len(functions)} function chunks")
        print(f"   - {len(classes)} class chunks")
        
        return all_chunks

# Test it
if __name__ == "__main__":
    chunker = GraphChunker()
    chunks = chunker.get_all_chunks()
    
    # Print first chunk as example
    if chunks:
        print("\n📝 Example chunk:")
        print("=" * 50)
        print(chunks[0]['text'])
        print("=" * 50)
        print(f"Metadata: {chunks[0]['metadata']}")
    
    chunker.close()