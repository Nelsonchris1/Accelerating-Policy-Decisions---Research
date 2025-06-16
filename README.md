# COPGPT - RAG-based Policy Recommendation Chatbot

A sophisticated Retrieval-Augmented Generation (RAG) chatbot built with FastAPI, LangChain, and OpenAI, designed to provide intelligent policy recommendations and sustainability insights with a focus on environmental topics and carbon emissions.

## Features

- **Hybrid Search Architecture**: Combines FAISS vector search with Google Search fallback for comprehensive information retrieval
- **Conversational Memory**: Maintains context across multiple interactions for coherent conversations
- **Document Processing**: Supports multiple file formats (PDF, DOCX, TXT, CSV, XLSX, HTML, MD, PPT)
- **Metadata Preservation**: Retains source information and references for all retrieved documents
- **Real-time Web Search**: Falls back to Google Search when local knowledge base lacks information
- **Clean Response Formatting**: Provides well-structured responses with proper references
- **FastAPI Backend**: High-performance asynchronous API endpoints

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Web Interface │────▶│  FastAPI Server  │────▶│  RAG Pipeline   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                ┌──────────────────────────┴───────────────────────────┐
                                │                                                      │
                        ┌───────▼────────┐                                    ┌────────▼────────┐
                        │  FAISS Vector  │                                    │  Google Search  │
                        │     Store      │                                    │   (Fallback)    │
                        └────────────────┘                                    └─────────────────┘
```

## Prerequisites

- Python 3.8+
- OpenAI API Key
- Google Serper API Key (for web search functionality)
- FAISS-compatible system

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd COP29_RAG_Chatbot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   SERPER_API_KEY=your_serper_api_key_here
   ```

## Project Structure

```
COP29_RAG_Chatbot/
│
├── app.py                 # FastAPI application entry point
├── retriever.py          # Main RAG pipeline and chat logic
├── embeddings.py         # Document embedding and vector store management
├── file_loader.py        # Multi-format document loader
├── metadata.py           # Metadata inspection utilities
├── requirements.txt      # Python dependencies
│
├── models/               # Data models
│   └── index.py         # Chat model definitions
│
├── templates/            # HTML templates
│   └── index.html       # Chat interface
│
├── static/              # Static assets (CSS, JS, images)
│
└── test2_db/            # FAISS vector database storage
    └── document_chunks111/
```

## Configuration

### Vector Database Setup

1. **Prepare your documents**
   Place your documents in a folder for processing.

2. **Generate embeddings**
   ```bash
   python embeddings.py
   ```
   Follow the prompts to specify your document folder path.

3. **Update database path**
   Ensure the `db_path` in `retriever.py` points to your FAISS database:
   ```python
   db_path = r"path/to/your/faiss_db"
   ```

### Embedding Model

The system uses OpenAI's `text-embedding-3-large` model. You can modify this in `embeddings.py`:
```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
```

## Running the Application

1. **Start the FastAPI server**
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Access the chatbot**
   Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

## Usage

### Web Interface
- Type your questions in the chat interface
- The bot will search its knowledge base first
- If needed, it will perform web searches for current information
- References are provided for all responses

### API Endpoint
Send POST requests to `/chat`:
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is carbon neutrality?"}'
```

### Supported Queries
- Environmental policy questions
- Carbon emission inquiries
- Sustainability best practices
- COP29-related information
- General conversational queries

## Key Components

### Retriever Pipeline (`retriever.py`)
- Manages the hybrid search strategy
- Maintains conversation history
- Handles query preprocessing and response formatting

### Document Processing (`file_loader.py`)
- Supports multiple file formats
- Preserves metadata during loading
- Implements fallback loaders for reliability

### Vector Store (`embeddings.py`)
- Creates and manages FAISS indexes
- Handles document chunking with overlap
- Preserves metadata through the embedding process

## Development

### Adding New Document Types

Extend the `FILE_LOADER_MAPPING` in `file_loader.py`:
```python
FILE_LOADER_MAPPING = {
    ".new_ext": (YourLoaderClass, {"param": "value"}),
    # ... existing mappings
}
```

### Customizing Responses

Modify the `predefined_responses` dictionary in `retriever.py` to add custom responses for common queries.

### Adjusting Search Parameters

Configure search behavior in `hybrid_chain()`:
- `k=5`: Number of documents to retrieve
- `chunk_size=1000`: Size of text chunks
- `chunk_overlap=100`: Overlap between chunks

## Troubleshooting

### Common Issues

1. **FAISS Loading Errors**
   - Ensure `allow_dangerous_deserialization=True` is set
   - Check file permissions on the database directory

2. **API Key Issues**
   - Verify `.env` file is in the root directory
   - Check API key validity

3. **Memory Issues**
   - Reduce chunk size or number of retrieved documents
   - Consider using a smaller embedding model

### Debug Mode

Enable detailed logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Performance Optimization

- **Async Processing**: FastAPI handles requests asynchronously
- **Caching**: Consider implementing Redis for response caching
- **Batch Processing**: Process multiple documents simultaneously
- **Index Optimization**: Regularly rebuild FAISS indexes for optimal performance

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://langchain.com/) for RAG capabilities
- Powered by [OpenAI](https://openai.com/) for embeddings and language models
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework

## Contact

For questions or support, please contact: info@carbonnote.ai

## Contributors

1.	Elizabeth Osanyinro, University of Bradford, UK

2.	Oluwole Fagbohun, Carbonnote, USA

3.	Ernest Effiong Offiong, Carbonnote, USA

4.	Maxwell Nwanna, RideNear, UK

5.	Grace Farayola, University of Buckingham, UK

6.	Olaitan Olaonipekun, Vuhosi Limited, UK

7.	Abiola Oludotun, Readrly Limited, UK

8.	Sayo Agunbiade, Independent Researcher, UK

9.	Oladotun Fasogbon, Independent Researcher, UK

10.	Ogheneruona Maria Esegbon-Isikeh, Readrly Limited, UK

11.	Lanre Shittu, Independent Researcher, UK

12.	Toyese Oloyede, Independent Researcher, UK

13.	Sa'id Olanrewaju, Readrly Limited, UK

14.	Christopher J Ozurumba, Independent Researcher, UK


**Note**: This is a beta version. For production use, please ensure proper security measures, rate limiting, and error handling are implemented.

