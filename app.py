import streamlit as st
import tempfile

from modules.pdf_loader import load_pdf
from modules.text_splitter import split_documents
from modules.embeddings import create_embeddings
from modules.vector_store import create_vector_store
from modules.retriever import get_retriever
from modules.qa_chain import create_qa_chain

st.title("AI PDF Question Answering System")

st.write("Upload a PDF and ask questions about the document.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:

    st.write("Processing PDF...")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    try:
        # Load PDF
        documents = load_pdf(file_path)
        st.success("PDF loaded")

        # Split text
        chunks = split_documents(documents)
        st.success("Text split into chunks")

        # Create embeddings
        embeddings = create_embeddings()
        st.success("Embeddings model loaded")

        # Create vector store
        vector_store = create_vector_store(chunks, embeddings)
        st.success("Vector database created")

        # Create retriever
        retriever = get_retriever(vector_store)

        # Create QA chain
        qa_chain = create_qa_chain(retriever)

        question = st.text_input("Ask a question from the PDF")

        if question:
            answer = qa_chain.invoke({"query": question})
            st.write("### Answer")
            st.write(answer)

    except Exception as e:
        st.error(f"Error occurred: {e}")