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
import getpass


# Load environment variables
load_dotenv()

# Streamlit UI
st.title("PDF Question Answering Application")
st.write("Upload a PDF and ask questions about it.")

# File upload section
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    # Save the uploaded PDF file
    with open("uploaded_pdf.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and process the PDF
    loader = PyPDFLoader("uploaded_pdf.pdf")
    data = loader.load()

    # Show the document details
    # st.write(f'You have {len(data)} document(s) in your data.')
    # st.write(f'There are {len(data[0].page_content)} characters in your sample document.')
    # st.write(f'Here is a sample: {data[0].page_content[:200]}')

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
    texts = text_splitter.split_documents(data)

    # Show the number of chunks
    #st.write(f'Now you have {len(texts)} documents (chunks).')

    # Set up the embeddings and vector store
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] 

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("Explain how AI works")

    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass(GOOGLE_API_KEY)


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

    # User question input
    query = st.text_input("Ask a question about the document:")
    if query:
        # Retrieve relevant documents from the vector store
        results = vector_store.similarity_search(query, k=5)
        # st.write("Relevant documents:")
        # for res in results:
        #     st.write(f"* {res.page_content} [{res.metadata}]")
        
        #Generate the answer using Google Generative AI
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(
            f"For the given query:{query}, generate a short answer based on the results:{results}. If the relevant information is not available, return as non without making things up."
        )
        st.write("Answer: ")
        st.write(response.text)
