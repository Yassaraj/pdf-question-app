import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from uuid import uuid4

# Load environment variables
load_dotenv()

# Streamlit UI
st.title("PDF Question Answering Application")
st.write("Upload a PDF and ask questions about its content.")

# File upload section
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    st.write("Processing your file, please wait...")

    # Show a progress bar while loading the PDF
    progress_bar = st.progress(0)
    progress = 0

    # Save the uploaded PDF file
    with open("uploaded_pdf.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    progress += 20
    progress_bar.progress(progress)

    # Load and process the PDF
    loader = PyPDFLoader("uploaded_pdf.pdf")
    data = loader.load()
    progress += 20
    progress_bar.progress(progress)

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
    texts = text_splitter.split_documents(data)
    progress += 20
    progress_bar.progress(progress)

    # Set up the embeddings and vector store
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    uuids = [str(uuid4()) for _ in range(len(texts))]
    vector_store.add_documents(documents=texts, ids=uuids)
    progress += 40
    progress_bar.progress(progress)

    st.success("PDF successfully processed!")

    # Chat-like user input for querying the document
    st.write("You can now ask questions about the uploaded PDF.")
    query = st.chat_input("Ask a question:")
    if query:
        with st.spinner("Fetching the answer..."):
            # Retrieve relevant documents from the vector store
            results = vector_store.similarity_search(query, k=5)

            # Generate the answer using Google Generative AI
            response = genai.GenerativeModel("gemini-1.5-flash").generate_content(
                f"For the given query: {query}, generate a short answer based on the results: {results}. "
                "If the relevant information is not available, return 'No relevant information found.' without making things up."
            )

        # Display the result
        st.chat_message("user").write(query)
        st.chat_message("assistant").write(response.text)
