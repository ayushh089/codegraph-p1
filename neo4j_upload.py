from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

class Neo4jUploader:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )
    
    def close(self):
        self.driver.close()
    
    def clear_database(self):
        """Remove all existing nodes (for fresh start)"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("🗑️  Cleared existing database")
    
    def create_nodes(self, functions: dict, classes: dict, files: set):
        """Create Function, Class, and File nodes"""
        with self.driver.session() as session:
            # Create Function nodes
            for func_name, file_path in functions.items():
                session.run(
                    "MERGE (f:Function {name: $name, file: $file})",
                    name=func_name, file=file_path
                )
            print(f"  ✅ Created {len(functions)} Function nodes")
            
            # Create Class nodes
            for class_name, file_path in classes.items():
                session.run(
                    "MERGE (c:Class {name: $name, file: $file})",
                    name=class_name, file=file_path
                )
            print(f"  ✅ Created {len(classes)} Class nodes")
            
            # Create File nodes (NEW)
            for file_path in files:
                session.run(
                    "MERGE (f:File {path: $path})",
                    path=file_path
                )
            print(f"  ✅ Created {len(files)} File nodes")
    
    def create_relationships(self, calls: list, class_contains: list, imports: list):
        """Create CALLS, CONTAINS, and IMPORTS relationships"""
        with self.driver.session() as session:
            # Create CALLS relationships
            for caller, callee, file_path in calls:
                session.run(
                    """
                    MATCH (caller:Function {name: $caller})
                    MATCH (callee:Function {name: $callee})
                    MERGE (caller)-[:CALLS {file: $file}]->(callee)
                    """,
                    caller=caller, callee=callee, file=file_path
                )
            print(f"  ✅ Created {len(calls)} CALLS relationships")
            
            # Create CONTAINS relationships (Class -> Function)
            for class_name, func_name, file_path in class_contains:
                session.run(
                    """
                    MATCH (c:Class {name: $class_name})
                    MATCH (f:Function {name: $func_name})
                    MERGE (c)-[:CONTAINS {file: $file}]->(f)
                    """,
                    class_name=class_name, func_name=func_name, file=file_path
                )
            print(f"  ✅ Created {len(class_contains)} CONTAINS relationships")
            
            # Create IMPORTS relationships (NEW)
            for from_file, to_module, line_no in imports:
                session.run(
                    """
                    MATCH (f1:File {path: $from_file})
                    MERGE (f2:File {path: $to_module})
                    MERGE (f1)-[:IMPORTS {line: $line}]->(f2)
                    """,
                    from_file=from_file, to_module=to_module, line=line_no
                )
            print(f"  ✅ Created {len(imports)} IMPORTS relationships")
    
    def create_file_contains_relationships(self, file_to_functions: list, file_to_classes: list):
        """Create CONTAINS relationships from File to Function and File to Class"""
        with self.driver.session() as session:
            # File contains Function
            for file_path, func_name in file_to_functions:
                session.run(
                    """
                    MATCH (f:File {path: $file_path})
                    MATCH (fn:Function {name: $func_name})
                    MERGE (f)-[:CONTAINS]->(fn)
                    """,
                    file_path=file_path, func_name=func_name
                )
            print(f"  ✅ Created {len(file_to_functions)} FILE-CONTAINS-FUNCTION relationships")
            
            # File contains Class
            for file_path, class_name in file_to_classes:
                session.run(
                    """
                    MATCH (f:File {path: $file_path})
                    MATCH (c:Class {name: $class_name})
                    MERGE (f)-[:CONTAINS]->(c)
                    """,
                    file_path=file_path, class_name=class_name
                )
            print(f"  ✅ Created {len(file_to_classes)} FILE-CONTAINS-CLASS relationships")
    
    def verify_graph(self):
        """Show graph statistics"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n) 
                RETURN labels(n)[0] as type, count(n) as count
                ORDER BY count DESC
            """)
            
            print("\n📊 Graph Statistics:")
            for record in result:
                print(f"   - {record['type']}: {record['count']} nodes")
            
            result = session.run("""
                MATCH ()-[r]->() 
                RETURN type(r) as relationship, count(r) as count
            """)
            
            print("\n🔗 Relationships:")
            for record in result:
                print(f"   - {record['relationship']}: {record['count']} edges")