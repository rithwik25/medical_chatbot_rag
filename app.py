from flask import Flask, request, jsonify, render_template
import os
import asyncio
from typing import List, Dict, Any
from data_models import Document, Chunk, RAGResponse
from rag_pipeline import RAGPipeline
from logger import logger
from asgiref.wsgi import WsgiToAsgi
import uvicorn

app = Flask(__name__)
asgi_app = WsgiToAsgi(app)  # Convert WSGI app to ASGI

# Initializing the RAG pipeline
rag = None

# Initialize RAG pipeline just once at startup
def initialize_rag():
    global rag
    logger.info("Initializing RAG pipeline...")
    rag = RAGPipeline(
        embedding_model_type="openai",
        embedding_model_name="text-embedding-3-small",
        llm_model="gpt-4o-mini",
        use_hybrid_search=True
    )
    
    # Load saved state if it exists, otherwise ingest sample documents
    if os.path.exists("./medical_rag_model"):
        logger.info("Loading saved RAG pipeline state...")
        rag = RAGPipeline.load_state("./medical_rag_model")
    else:
        logger.info("Loading sample documents...")
        rag.ingest_documents(load_sample=True)
        logger.info("Saving RAG pipeline state...")
        rag.save_state("./medical_rag_model")
    
    logger.info("RAG pipeline initialized successfully")

@app.route('/')
def index():
    """Render the main page with a simple UI for querying the RAG pipeline."""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
async def query():
    """Process a query against the RAG pipeline."""
    if rag is None:
        return jsonify({"error": "RAG pipeline not initialized"}), 500
    
    data = request.json
    query_text = data.get('query')
    top_k = data.get('top_k', 3)
    
    if not query_text:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        logger.info(f"Processing query: {query_text}")
        response = await rag.query_async(query_text, top_k=top_k)
        
        # Return both structured data and formatted response
        result = {
            "answer": response.answer,
            "references": response.references,
            "followup_questions": response.followup_questions,
            "disclaimer": response.disclaimer,
            "formatted_response": getattr(response, 'formatted_response', None)
        }
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500

@app.route('/api/metrics', methods=['GET'])
def metrics():
    """Get performance metrics for the RAG pipeline."""
    if rag is None:
        return jsonify({"error": "RAG pipeline not initialized"}), 500
    
    metrics_data = rag.get_metrics()
    return jsonify(metrics_data)

# Initialize RAG before running the server
initialize_rag()

# For running with uvicorn directly(due to async methods)
if __name__ == "__main__":
    # Run with Uvicorn
    uvicorn.run(
        "app:asgi_app",
        host="0.0.0.0",
        port=5000,
        log_level="info",
        reload=True  # Enables auto-reload during development
    )

    # Alternative for running with Hypercorn
    # import hypercorn.asyncio
    # hypercorn.asyncio.run(asgi_app, hypercorn.Config(
    #     bind=["0.0.0.0:5000"],
    #     accesslog="-"
    # ))