# Medical RAG Chatbot

A Retrieval Augmented Generation (RAG) system designed specifically for medical information retrieval and question answering.

## Features

- Hybrid search combining dense vector retrieval (FAISS) and sparse retrieval (BM25)
- Hierarchical document chunking for improved context retention
- Asynchronous processing for improved performance
- Embedding caching for faster retrieval
- LLM-powered response generation with proper citations
- Web-based user interface

## Architecture

The system is built with a modular architecture consisting of:

1. **Data Ingestion & Processing**: Hierarchical chunking of medical documents
2. **Embedding Generation**: OpenAI embeddings with caching
3. **Vector Store**: Hybrid search combining FAISS and BM25
4. **Retrieval**: Intelligent retrieval with metadata filtering
5. **LLM Integration**: Response generation with OpenAI models
6. **Web Interface**: Flask-based UI for interacting with the system

## Prerequisites

- Docker and Docker Compose
- OpenAI API key

## Setup & Installation

1. Clone the repository
   ```
   git clone <repository-url>
   cd medical-rag-chatbot
   ```

2. Set your OpenAI API key in an environment variable
   ```
   export OPENAI_API_KEY=your_openai_api_key
   ```

3. Build and start the Docker container
   ```
   docker-compose up -d
   ```

4. Access the application at http://localhost:5000

## Usage

1. Enter a medical question in the search box
2. View the AI-generated answer with source citations
3. Explore suggested follow-up questions

## Project Structure

```
medical-rag-chatbot/
├── app.py                     # Flask application
├── data_models.py             # Data structures
├── embedding_generator.py     # Embedding generation
├── data_processing.py         # Document processing
├── vector_store.py            # FAISS and BM25 hybrid search
├── retriever.py               # Chunk retrieval
├── llm_integration.py         # LLM interaction
├── rag_pipeline.py            # End-to-end pipeline
├── logger.py                  # Logging configuration
├── templates/                 # HTML templates
│   └── index.html            # Main UI
├── Dockerfile                 # Docker configuration
└── docker-compose.yml         # Docker Compose configuration
```

## Development

To run the application in development mode:

```
docker-compose up
```

This will enable auto-reloading of the application on code changes.

## Performance Metrics

The system tracks various performance metrics:
- Average retrieval time
- Average generation time
- Cache hit rate
- Total queries processed

View these metrics at the `/api/metrics` endpoint.