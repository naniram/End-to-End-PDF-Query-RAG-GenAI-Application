import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader  # Importing PdfReader for PDF text extraction
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text.strip()
    except Exception as e:
        print(f"An error occurred while extracting text from PDF: {e}")
        return None

# Function to create a vector database from text
def create_vector_db(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    return vector_store

# Function to set up the RAG chain
def setup_rag_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    rag_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = vector_store.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents = True
    )

    return rag_chain
