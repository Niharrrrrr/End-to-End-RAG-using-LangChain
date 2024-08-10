import streamlit as st
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Define the function to handle user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    # Display the output in a text area for easy copying
    st.text_area("Generated Response:", value=response["output_text"], height=200)

# Define the main function for the Streamlit app
def main():
    st.set_page_config(page_title="Chat PDF with Gemini", page_icon="💁", layout="wide")
    
    # Header Section
    st.markdown("""
    <style>
        .main-header { 
            font-size: 2.5rem; 
            font-weight: 600; 
            color: #333; 
            text-align: center;
            margin-top: 20px;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #666;
            text-align: center;
            margin-bottom: 20px;
        }
        .sidebar .sidebar-content {
            padding: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='main-header'>Chat with PDF using Gemini💁</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Upload your PDF, ask questions, and get instant answers!</div>", unsafe_allow_html=True)

    # User Question Input
    user_question = st.text_input(
        "Ask a Question from the PDF Files",
        placeholder="Type your question here...",
        label_visibility="collapsed"
    )

    if user_question:
        with st.spinner("Generating response..."):
            user_input(user_question)

    # Sidebar for File Upload
    with st.sidebar:
        st.title("Menu")
        st.markdown("<p style='color: #666; margin-top: -10px;'>Upload your PDF and process it to start chatting.</p>", unsafe_allow_html=True)
        
        pdf_docs = st.file_uploader(
            "Upload PDF Files",
            accept_multiple_files=True,
            help="You can upload multiple PDF files.",
            type=["pdf"]
        )
        
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Files processed successfully!")
            else:
                st.warning("Please upload at least one PDF file.")

# Run the main function
if __name__ == "__main__":
    main()
