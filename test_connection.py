from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

def test_connection():
    try:
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )
        
        with driver.session() as session:
            result = session.run("RETURN 'Connected to Aura!' as message")
            print(result.single()["message"])
            print(f"✅ Successfully connected to {os.getenv('NEO4J_URI')}")
        
        driver.close()
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your URI (should start with neo4j+s://)")
        print("2. Verify username (usually 'neo4j')")
        print("3. Double-check password")
        print("4. Make sure your Aura instance is running (green dot in console)")

if __name__ == "__main__":
    test_connection()