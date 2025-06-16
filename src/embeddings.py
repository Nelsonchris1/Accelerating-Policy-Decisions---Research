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

# Check if the key is loaded
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    logger.error("OpenAI API Key not found in environment variables.")
else:
    logger.info("OpenAI API Key loaded successfully.")

def document_loader(folder_path):
    """
    Wrapper function to load documents using file_loader.py and retain metadata.
    """
    logger.info(f"Loading documents from folder: {folder_path}")
    try:
        documents = load_document(folder_path)  # Load documents with metadata
        logger.debug(f"Loaded {len(documents)} documents from {folder_path}")

        # Split documents into chunks while retaining metadata
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        all_splits = []
        for doc in documents:
            splits = text_splitter.split_text(doc.page_content)
            for split in splits:
                chunk = Document(page_content=split, metadata=doc.metadata)  # Retain metadata
                print(f"Chunk created with metadata: {chunk.metadata}")  # Debug log
                all_splits.append(chunk)


        logger.info(f"Split documents into {len(all_splits)} chunks.")
        return all_splits
    except Exception as e:
        logger.error(f"Error loading or processing documents from folder {folder_path}: {e}")
        raise


def save_splits_to_txt(all_splits, output_file):
    """
    Saves document chunks with metadata to a text file with a delimiter.
    """
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in all_splits:
                # Create a dictionary to save both content and metadata
                data = {
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata
                }
                f.write(json.dumps(data) + "\n---\n")  # Save as JSON string
        logger.info(f"Document chunks with metadata saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving document chunks to file: {e}")
        raise


def load_splits_from_txt(input_file):
    """
    Loads document chunks with metadata from a text file.
    """
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = content.split("\n---\n")
        
        # Parse JSON data and create Document objects
        documents = [
            Document(**json.loads(chunk)) for chunk in chunks if chunk.strip()
        ]
        logger.info(f"Loaded {len(documents)} document chunks with metadata from {input_file}")
        return documents
    except Exception as e:
        logger.error(f"Error loading document chunks from file: {e}")
        raise


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

# def vector_db(all_splits, embeddings, persist_directory):
#     """
#     Creates a FAISS vector database.
#     """
#     logger.info("Creating FAISS vector database...")
#     try:
#         if not all_splits:
#             raise ValueError("Document splits are empty.")
#         if not embeddings:
#             raise ValueError("Embeddings are not initialized.")
#         if not os.access(persist_directory, os.W_OK):
#             raise PermissionError(f"Cannot write to directory: {persist_directory}")
#         faiss_db = FAISS.from_documents(all_splits, embeddings)
#         faiss_db.save_local(persist_directory)
#         logger.info(f"FAISS vector database saved to {persist_directory}")
#         return faiss_db
#     except Exception as e:
#         logger.error(f"Error creating or saving FAISS vector database: {e}")
#         raise


def vector_db(all_splits, embeddings, persist_directory):
    """
    Creates a FAISS vector database and saves metadata.
    """
    logger.info("Creating FAISS vector database...")
    try:
        if not all_splits:
            raise ValueError("Document splits are empty.")
        if not embeddings:
            raise ValueError("Embeddings are not initialized.")
        if not os.access(persist_directory, os.W_OK):
            raise PermissionError(f"Cannot write to directory: {persist_directory}")

        # Generate unique IDs for documents
        uuids = [str(uuid4()) for _ in range(len(all_splits))]

        # Create FAISS vector store and add documents
        faiss_db = FAISS.from_documents(all_splits, embeddings)
        # faiss_db.add_documents(documents=all_splits, ids=uuids)

        # Save the vector store locally
        faiss_db.save_local(persist_directory)
        logger.info(f"FAISS vector database saved to {persist_directory}")
        return faiss_db
    except Exception as e:
        logger.error(f"Error creating or saving FAISS vector database: {e}")
        raise




def load_faiss_vectorstore(db_path, embeddings):
    """
    Load a FAISS vector store from the given path with safe deserialization.
    """
    try:
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"FAISS database directory not found: {db_path}")
        
        # Allow safe deserialization for trusted files
        vectorstore = FAISS.load_local(
            db_path,
            embeddings,
            allow_dangerous_deserialization=True  # Enable this if you trust the source
        )
        logger.info("FAISS vector store loaded successfully.")
        return vectorstore
    except Exception as e:
        logger.error(f"Error loading FAISS vector store: {e}")
        raise

    
def inspect_faiss_metadata(db_path, embeddings):
    """
    Inspect metadata stored in the FAISS vector store.
    """
    try:
        # Load FAISS vector store with deserialization allowed
        vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        
        # Iterate over the internal dictionary of documents
        for idx, doc in enumerate(vectorstore.docstore._dict.values(), start=1):
            print(f"Document {idx}:")
            print(f"Content: {doc.page_content[:100]}...")  # Preview first 100 characters
            print(f"Metadata: {doc.metadata}")  # Display metadata
            logger.debug(f"Retrieved document metadata: {doc.metadata}")
    except Exception as e:
        logger.error(f"Error inspecting FAISS metadata: {e}")
        raise


if __name__ == "__main__":
    # folder_path = input("Enter the folder path containing documents: ").strip()
    persist_directory = r"test2_db\document_chunks111"
    # output_file = r"document_chunks111.txt"

    # if not os.path.exists(folder_path):
        # logger.error(f"Invalid folder path: {folder_path}")
        # exit(1)

    os.makedirs(persist_directory, exist_ok=True)

    try:
        # Load and split documents
        # all_splits = document_loader(folder_path)
        
        # Save splits to a file
        # save_splits_to_txt(all_splits, output_file)
        
        # Load splits from the saved file
        # all_splits = load_splits_from_txt(output_file)
        # print(all_splits)
        
        # Initialize embeddings and create vector database
        embeddings = embedding_model()
        
        # Measure time for embedding
        start_time = time.time()
        # vector_db(all_splits, embeddings, persist_directory)
        elapsed_time = time.time() - start_time
        logger.info(f"Time taken to create embeddings: {elapsed_time:.2f} seconds")
        
        # Inspect FAISS metadata
        logger.info("Inspecting FAISS metadata...")
        inspect_faiss_metadata(persist_directory, embeddings)

        logger.info("FAISS vector database creation and inspection completed successfully.")

    except Exception as e:
        logger.critical(f"Process failed: {e}")
