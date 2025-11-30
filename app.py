import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from transformers import pipeline
import tempfile

st.title("📘 Local Knowledge Base AI Agent")

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded:
    # Save uploaded file to a temporary path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        temp_path = tmp.name

    # Load PDF from temp file
    loader = PyPDFLoader(temp_path)
    docs = loader.load()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vector Database
    vectordb = Chroma.from_documents(chunks, embedding=embeddings)
    retriever = vectordb.as_retriever()

    # Local text-generation model
    model = pipeline("text-generation", model="distilgpt2")

    st.success("PDF Loaded & Indexed Successfully! ✔")

    query = st.text_input("Ask a question from your PDF:")

    if query:
        # Retrieve relevant chunks
        results = retriever.invoke(query)

        context = " ".join([d.page_content for d in results])

        prompt = (
            f"Use ONLY this context to answer:\n\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )

        answer = model(prompt, max_length=200)[0]['generated_text']

        st.write("### 🧠 Answer:")
        st.write(answer)
