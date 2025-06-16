
import os
import logging
from file_loader import load_document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import time
from langchain_core.documents import Document
import faiss
from uuid import uuid4
from langchain_community.docstore.in_memory import InMemoryDocstore
import json

# Load the .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def embedding_model():
    """
    Initializes the OpenAI embedding model.
    """
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        logger.debug("OpenAI embedding model initialized successfully.")
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing OpenAI embedding model: {e}")
        raise

# Initialize embeddings and create vector database
embeddings = embedding_model()
        
# Load FAISS store
faiss_store = FAISS.load_local(r"test2_db/document_chunks111", embeddings, allow_dangerous_deserialization=True)

# Perform similarity search
query = "Katowice Committee "
retrieved_docs = faiss_store.similarity_search(query, k=1)

# print(retrieved_docs)

# Deserialize metadata from page_content
for doc in retrieved_docs:
    # Parse the page_content as JSON
    content = json.loads(doc.page_content)
    actual_content = content.get("page_content", "No content available")
    metadata = content.get("metadata", {})
    source = metadata.get("source", "Unknown source")

    # Display the results
    print(f"Content: {actual_content}")
    print(f"Metadata: {source}")