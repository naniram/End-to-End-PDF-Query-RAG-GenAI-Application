import streamlit as st
from src.utils import extract_text_from_pdf, create_vector_db, setup_rag_chain

st.header("PDF Query System")

# Initialize session state variables for the PDF and vector store
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:    
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if st.button("Process PDF") and pdf_file:
        with st.spinner('Processing...'):
            # Extract text from the uploaded PDF
            pdf_text = extract_text_from_pdf(pdf_file)
            
            # Create vector database from the extracted text
            st.session_state.vector_store = create_vector_db(pdf_text)
            st.session_state.pdf_text = pdf_text
            st.success("Done")

if st.session_state.vector_store:
    # Set up the RAG chain
    rag_chain = setup_rag_chain(st.session_state.vector_store)
    
    query = st.text_input("Enter your query")
    if query:
        # Get the answer
        result = rag_chain({"query": query})
        st.write(result['result'])
