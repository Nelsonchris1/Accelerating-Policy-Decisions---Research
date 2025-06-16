import os
import re
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import GoogleSerperAPIWrapper

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv()

# Initialize APIs
serper_api_key = os.getenv("SERPER_API_KEY")
serper = GoogleSerperAPIWrapper(api_key=serper_api_key)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Functions
def clean_text(text):
    """Remove special characters from the text."""
    return re.sub(r'[^A-Za-z0-9\s.,]', '', text)

def load_faiss_vectorstore(db_path, embeddings):
    """Load FAISS vector store with safe deserialization."""
    return FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

def combine_docs(docs):
    """Combine retrieved document contents into a single string."""
    return "\n\n".join([doc.page_content for doc in docs])

def extract_references(docs):
    """Extract and format unique references from document metadata or embedded JSON in page_content."""
    references = set()  # Use a set to ensure uniqueness
    for doc in docs:
        # Attempt to parse metadata directly
        metadata = doc.metadata
        
        # If metadata is empty, attempt to extract it from the page_content
        if not metadata:
            try:
                content = json.loads(doc.page_content)  # Parse the page_content as JSON
                metadata = content.get("metadata", {})
                
            except json.JSONDecodeError:
                metadata = {}

        # Extract the source if available
        if "source" in metadata:
            source = metadata["source"]
            page = metadata.get("page", "N/A")  # Add page number if available
            references.add(f"Source: {source}")
    
    if not references:
        return "No references available."
    return "\n".join(sorted(references))  # Sort references for consistency


def google_search(query):
    """Perform a Google search using SerperDev."""
    response = serper.run(query)
    # print(response)
    # if "results" in response:
    #     return response["results"]
    # return []
    return response

# def extract_search_content(results):
#     """Extract relevant content from Google Search results."""
#     content = []
#     for result in results:
#         title = result.get("title", "No Title")
#         snippet = result.get("snippet", "No Snippet")
#         link = result.get("link", "No Link")
#         content.append(f"{title}\n{snippet}\n{link}")
#     return "\n\n".join(content)

def generate_recommendations(context, query):
    """Generate recommendations using the synthesis agent."""
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Synthesis Agent specialized in turning raw data into actionable policy insights. "
                "Use the provided context and query to generate recommendations.",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuery: {query}\n\n"
                "Answer concisely and include actionable recommendations with explanations."
            ),
        ]
    )
    formatted_prompt = prompt_template.format(context=context, query=query)
    return llm.invoke([formatted_prompt]).content.strip()

def hybrid_chain(query, db_path, embeddings, fallback_google=True):
    """Perform FAISS search and fallback to Google Search if needed."""
    try:
        vectorstore = load_faiss_vectorstore(db_path, embeddings)
        retrieved_docs = vectorstore.similarity_search(query, k=5)
        
        if retrieved_docs:
            print(f"Retrieved {len(retrieved_docs)} documents from FAISS.")
            faiss_context = combine_docs(retrieved_docs)
            references = extract_references(retrieved_docs)
        else:
            print("No relevant documents found in FAISS.")
            faiss_context = ""
            references = "No references available."
        
        if not retrieved_docs and fallback_google:
            print("Falling back to Google Search...")
            search_results = google_search(query)
            print(search_results)
            if search_results:
                faiss_context += "\n\n" + search_results  
            else:
                print("No results found in Google Search.")
                return "No relevant context found.", references
            
        # Step 3: Generate Recommendations
        response = generate_recommendations(faiss_context.strip(), query)
        return response, references
    
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred during processing.", "No references available."


conversation_history = {}
# Main Function
def main(query, user_id="default_user"):
    db_path = r"D:\COP29_RAG_Chatbot\test2_db\document_chunks111"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    query_lower = query.lower().strip()

    if user_id not in conversation_history:
        conversation_history[user_id] = []

    # Handle common predefined queries
    predefined_responses = {
        "hi": "Hello! 😊 How can I assist you today?",
        "hello": "Hi there! 👋 What can I help you with?",
        "hey": "Hey! 👋 Need any assistance?",
        "good morning": "Good morning! ☀️ How can I assist you?",
        "good afternoon": "Good afternoon! 🌞 How may I help?",
        "good evening": "Good evening! 🌙 How can I assist?",
        "hola": "¡Hola! 😊 How may I assist you today?",
        
        "bye": "Goodbye! 👋 Have a great day!",
        "goodbye": "Take care! Hope to see you again soon! 😊",
        "see you": "See you later! 🚀 Feel free to return anytime.",
        
        "who are you": "I am COPGPT, your AI assistant, helping with policy recommendations. How can I assist?",
        "what is your name": "I’m COPGPT, your AI assistant for sustainability and carbon insights! 🌍",
        "what can you do": "I can help answer questions, provide environmental insights, and assist with sustainability projects. Just ask me anything! 😊",
        
        "thank you": "You're very welcome! 😊 Happy to help!",
        "thanks": "No problem! 👍 Glad to assist.",
        
        "sorry": "No worries! 😊 I'm here to help.",
        "my bad": "No problem at all! 🚀 We all make mistakes.",
        
        "you are great": "Thank you! That means a lot! 😊",
        "awesome bot": "You're awesome too! 💙 Happy to assist!",
        "you are smart": "I appreciate that! 🧠 I'm here to help!",
        "what is carbonnote": "CarbonNote is a tool for tracking and analyzing carbon emissions efficiently."
    }

    if query_lower in predefined_responses:
        return predefined_responses[query_lower]

    past_context = "\n".join(conversation_history[user_id][-3:]) if conversation_history[user_id] else ""

    contextual_query = f"Previous conversation:\n{past_context}\n\nUser's new query: {query}"
    result, references = hybrid_chain(contextual_query, db_path, embeddings)

    clean_result = clean_text(result.strip())

    formatted_response = "\n\n".join(clean_result.split(". "))  

    if references and "No references available" not in references:
        formatted_response += f"\n\n📌 References:\n{references}"

    if "context provided" not in formatted_response.lower():
        formatted_response += (
            "\n\n(If this response doesn't fully address your query, please contact info@carbonnote.ai for assistance.)"
        )

    conversation_history[user_id].append(f"User: {query}")
    conversation_history[user_id].append(f"Bot: {formatted_response}")

    conversation_history[user_id] = conversation_history[user_id][-10:]

    print(formatted_response)
    return formatted_response


def chatPrompt(query, user_id="default_user"):
    db_path = r"/home/ubuntu/app/database/final_faiss_db"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    query_lower = query.lower().strip()

    if user_id not in conversation_history:
        conversation_history[user_id] = []

    # Handle common predefined queries
    predefined_responses = {
        "hi": "Hello! 😊 How can I assist you today?",
        "hello": "Hi there! 👋 What can I help you with?",
        "hey": "Hey! 👋 Need any assistance?",
        "good morning": "Good morning! ☀️ How can I assist you?",
        "good afternoon": "Good afternoon! 🌞 How may I help?",
        "good evening": "Good evening! 🌙 How can I assist?",
        "hola": "¡Hola! 😊 How may I assist you today?",
        
        "bye": "Goodbye! 👋 Have a great day!",
        "goodbye": "Take care! Hope to see you again soon! 😊",
        "see you": "See you later! 🚀 Feel free to return anytime.",
        
        "who are you": "I am COPGPT, your AI assistant, helping with policy recommendations. How can I assist?",
        "what is your name": "I’m COPGPT, your AI assistant for sustainability and carbon insights! 🌍",
        "what can you do": "I can help answer questions, provide environmental insights, and assist with sustainability projects. Just ask me anything! 😊",
        
        "thank you": "You're very welcome! 😊 Happy to help!",
        "thanks": "No problem! 👍 Glad to assist.",
        
        "sorry": "No worries! 😊 I'm here to help.",
        "my bad": "No problem at all! 🚀 We all make mistakes.",
        
        "you are great": "Thank you! That means a lot! 😊",
        "awesome bot": "You're awesome too! 💙 Happy to assist!",
        "you are smart": "I appreciate that! 🧠 I'm here to help!",
        "what is carbonnote": "CarbonNote is a tool for tracking and analyzing carbon emissions efficiently."
    }

    if query_lower in predefined_responses:
        return predefined_responses[query_lower]

    past_context = "\n".join(conversation_history[user_id][-3:]) if conversation_history[user_id] else ""

    contextual_query = f"Previous conversation:\n{past_context}\n\nUser's new query: {query}"
    result, references = hybrid_chain(contextual_query, db_path, embeddings)

    clean_result = clean_text(result.strip())

    formatted_response = "\n\n".join(clean_result.split(". "))  

    if references and "No references available" not in references:
        formatted_response += f"\n\n📌 References:\n{references}"

    if "context provided" not in formatted_response.lower():
        formatted_response += (
            "\n\n(If this response doesn't fully address your query, please contact info@carbonnote.ai for assistance.)"
        )

    conversation_history[user_id].append(f"User: {query}")
    conversation_history[user_id].append(f"Bot: {formatted_response}")

    conversation_history[user_id] = conversation_history[user_id][-10:]

    print(formatted_response)
    return formatted_response



if __name__ == "__main__":
    main()
