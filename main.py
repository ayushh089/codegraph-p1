from parser import CodeParser
from neo4j_upload import Neo4jUploader
import sys
from pathlib import Path

def main():
    # Get repo path from command line or use default
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    else:
        repo_path = input("📁 Enter path to Python repo: ")
    
    if not Path(repo_path).exists():
        print(f"❌ Path {repo_path} does not exist")
        return
    
    print(f"\n🚀 Starting Phase 1: Code to Graph Pipeline")
    print(f"📂 Repository: {repo_path}\n")
    
    # Step 1: Parse code with AST
    print("=" * 50)
    print("STEP 1: Parsing Code with AST")
    print("=" * 50)
    
    parser = CodeParser()
    graph_data = parser.parse_repo(repo_path)
    
    # Step 2: Upload to Neo4j
    print("\n" + "=" * 50)
    print("STEP 2: Uploading to Neo4j")
    print("=" * 50)
    
    uploader = Neo4jUploader()
    
    # Clear existing data
    uploader.clear_database()
    
    # Create nodes (including File nodes)
    uploader.create_nodes(
        graph_data['functions'], 
        graph_data['classes'],
        graph_data['files']  # NEW: pass files
    )
    
    # Create relationships (including IMPORTS)
    uploader.create_relationships(
        graph_data['calls'], 
        graph_data['class_contains'],
        graph_data['imports']  # NEW: pass imports
    )
    
    # Create File CONTAINS relationships (NEW)
    uploader.create_file_contains_relationships(
        graph_data['file_to_functions'],
        graph_data['file_to_classes']
    )
    
    # Verify
    uploader.verify_graph()
    
    uploader.close()
    
    print("\n" + "=" * 50)
    print("✅ PHASE 1 COMPLETE!")
    print("=" * 50)
    print("\n📝 New queries to try in Neo4j Browser:")
    print("   1. MATCH (f:File) RETURN f LIMIT 10")
    print("   2. MATCH (f:File)-[:CONTAINS]->(func:Function) RETURN f.path, func.name")
    print("   3. MATCH (f1:File)-[:IMPORTS]->(f2:File) RETURN f1.path, f2.path")
    print("   4. MATCH path = (:File)-[:CONTAINS|CALLS*1..3]->() RETURN path LIMIT 20")

if __name__ == "__main__":
    main()