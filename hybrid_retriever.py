class HybridRetriever:
    """Combine Graph traversal + Vector search"""
    
    def __init__(self, neo4j_driver, vector_store):
        self.neo4j = neo4j_driver
        self.vector_store = vector_store
    
    def retrieve(self, query, top_k=5):
        """Hybrid search: Graph first, then Vector"""
        
        # Step 1: Extract entities from query
        entities = self._extract_entities(query)
        
        # Step 2: Graph traversal (exact relationships)
        graph_results = []
        for entity in entities:
            results = self._graph_search(entity)
            graph_results.extend(results)
        
        # Step 3: Vector search (semantic)
        vector_results = self.vector_store.search_similar(query, top_k)
        
        # Step 4: Merge results (graph results get higher weight)
        merged = self._merge_results(graph_results, vector_results)
        
        return merged
    
    def _extract_entities(self, query):
        """Extract function/class names from query"""
        import re
        # Pattern: "who calls validate_token" -> extract "validate_token"
        patterns = [
            r'calls (\w+)',
            r'called by (\w+)',
            r'function (\w+)',
            r'(\w+) function'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return [match.group(1)]
        return []
    
    def _graph_search(self, entity):
        """Search Neo4j for exact relationships"""
        with self.neo4j.session() as session:
            result = session.run("""
                MATCH (caller:Function)-[:CALLS]->(callee:Function {name: $name})
                RETURN caller.name as caller, caller.file as caller_file, 
                       callee.name as callee, callee.file as callee_file
            """, name=entity)
            
            return [dict(record) for record in result]
    
    def _merge_results(self, graph_results, vector_results):
        """Merge with priority to graph results"""
        merged = {
            'graph': graph_results,
            'vector': vector_results,
            'priority': 'graph_first'
        }
        return merged