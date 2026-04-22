import streamlit as st
from rag_pipeline import RAGPipeline
import sys
from io import StringIO

# Capture print output (since rag.ask() uses print statements)
class StreamlitOutput:
    def __init__(self):
        self.output = ""
    
    def write(self, text):
        self.output += text
    
    def get_output(self):
        return self.output

st.title("🔍 CodeGraph RAG")
st.markdown("Ask questions about your codebase in natural language!")

# Initialize RAG pipeline once
@st.cache_resource
def load_rag():
    with st.spinner("Loading RAG pipeline (this may take a moment)..."):
        return RAGPipeline()

try:
    rag = load_rag()
    st.success("✅ RAG pipeline ready!")
except Exception as e:
    st.error(f"❌ Error loading RAG pipeline: {e}")
    st.stop()

# Question input
question = st.text_input("💬 Ask about your codebase:", placeholder="e.g., which file calls get_chat_response")

if question:
    with st.spinner("🔍 Searching codebase..."):
        try:
            # Capture print output
            old_stdout = sys.stdout
            captured_output = StringIO()
            sys.stdout = captured_output
            
            # Get answer
            answer = rag.ask(question)
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Display answer
            st.markdown("### 💡 Answer")
            st.write(answer)
            
            # Also show any captured print output (for debugging)
            debug_output = captured_output.getvalue()
            if debug_output and st.checkbox("Show debug info"):
                with st.expander("🔍 Debug Output"):
                    st.code(debug_output)
                    
        except Exception as e:
            st.error(f"Error: {e}")