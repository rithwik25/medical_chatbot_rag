import os
import json
import time
import numpy as np
import asyncio
from typing import List, Dict, Any
from data_models import Document, Chunk, RAGResponse
from embedding_generator import EmbeddingGenerator
from data_processing import HierarchicalDataIngestion
from vector_store import HybridVectorStore
from retriever import Retriever
from llm_integration import LLMProcessor
from logger import logger

class RAGPipeline:
    """
    End-to-end RAG pipeline that integrates all components:
    1. Data ingestion and processing
    2. Embedding generation
    3. Vector storage and retrieval
    4. LLM response generation
    """
    
    def __init__(self, 
                embedding_model_type: str = "openai", 
                embedding_model_name: str = "text-embedding-3-small",
                llm_model: str = "gpt-4o-mini",
                use_hybrid_search: bool = True,
                cache_size: int = 100):
        """
        Initialize the RAG pipeline with configurable components.
        
        Args:
            embedding_model_type: Type of embedding model to use ('openai', 'huggingface', or 'medical')
            embedding_model_name: Name of the specific embedding model
            llm_model: Name of the LLM model to use
            use_hybrid_search: Whether to use hybrid search (vector + BM25)
            cache_size: Size of the embedding cache
        """
        # Initialize components
        self.data_ingestion = HierarchicalDataIngestion(max_chunk_size=500)
        self.embedding_generator = EmbeddingGenerator(model_type=embedding_model_type, model_name=embedding_model_name)
        
        # Vector store dimensions based on model type
        dimension = 1536 if embedding_model_type == "openai" else 768
        self.vector_store = HybridVectorStore(dimension=dimension)
        
        # Initialize the retriever with configurable parameters
        self.retriever = Retriever(
            embedding_generator=self.embedding_generator,
            vector_store=self.vector_store,
            use_hybrid_search=use_hybrid_search,
            cache_size=cache_size
        )
        
        # Initialize the LLM processor with the specified model
        self.llm_processor = LLMProcessor(model=llm_model)
        
        # Runtime metrics tracking
        self.metrics = {
            "total_queries": 0,
            "total_retrieval_time": 0,
            "total_generation_time": 0,
            "cache_hits": 0
        }
        
        logger.info(f"Initialized RAG pipeline with {embedding_model_type}/{embedding_model_name} embeddings and {llm_model}")
    
    def ingest_documents(self, documents: List[Document] = None, load_sample: bool = False) -> None:
        """
        Process documents through the pipeline.
        
        Args:
            documents: List of documents to process
            load_sample: Whether to load sample medical documents
        """
        if load_sample:
            documents = self.data_ingestion.load_sample_medical_documents()
        
        if not documents:
            logger.warning("No documents provided for ingestion")
            return
        
        # Chunk documents
        chunks = self.data_ingestion.chunk_documents(documents)
        
        # Generate embeddings
        embeddings = []
        chunk_ids = []
        
        for chunk in chunks:
            embedding = self.embedding_generator.generate_embedding(chunk.content)
            embeddings.append(embedding)
            chunk_ids.append(chunk.id)
        
        # Store embeddings
        self.vector_store.add_embeddings(np.array(embeddings), chunk_ids, chunks)
        
        logger.info(f"Successfully ingested {len(documents)} documents, created {len(chunks)} chunks")
    
    async def query_async(self, query: str, top_k: int = 5, filter_params: dict = None) -> RAGResponse:
        """
        Process a query through the RAG pipeline asynchronously.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            filter_params: Metadata filters to apply
            
        Returns:
            RAGResponse object with answer, references, and follow-up questions
        """
        self.metrics["total_queries"] += 1
        
        # Analyze the query for better retrieval
        query_info = self.retriever.analyze_query(query)
        
        # Retrieve relevant chunks
        retrieval_start = time.time()
        retrieved_chunks = await self.retriever.retrieve_async(query, k=top_k, filter_params=filter_params)
        retrieval_time = time.time() - retrieval_start
        self.metrics["total_retrieval_time"] += retrieval_time
        
        # Update cache hit metric if available from retriever
        if hasattr(self.retriever, "cache_hit") and self.retriever.cache_hit:
            self.metrics["cache_hits"] += 1
        
        # Generate response with the LLM
        generation_start = time.time()
        response_dict = await self.llm_processor.generate_response_async(query, retrieved_chunks)
        generation_time = time.time() - generation_start
        self.metrics["total_generation_time"] += generation_time
        
        # Format the response using the new formatter
        formatted_response = self.llm_processor.format_final_response(response_dict)
        
        # Create and return the RAG response with the properly formatted sections
        rag_response = RAGResponse(
            answer=response_dict["answer"],
            references=response_dict["references"],
            followup_questions=response_dict["followup_questions"],
            disclaimer=response_dict["disclaimer"],
            formatted_response=formatted_response  # Add the formatted response to the RAG response
        )
        
        # Log performance metrics
        logger.info(f"Query processed in {retrieval_time + generation_time:.2f}s "
                   f"(retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s)")
        
        return rag_response
    
    def query(self, query: str, top_k: int = 5, filter_params: dict = None) -> RAGResponse:
        """
        Process a query through the RAG pipeline (synchronous version).
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            filter_params: Metadata filters to apply
            
        Returns:
            RAGResponse object with answer, references, and follow-up questions
        """
        return asyncio.run(self.query_async(query, top_k, filter_params))
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the RAG pipeline.
        
        Returns:
            Dictionary of metrics
        """
        if self.metrics["total_queries"] > 0:
            avg_retrieval = self.metrics["total_retrieval_time"] / self.metrics["total_queries"]
            avg_generation = self.metrics["total_generation_time"] / self.metrics["total_queries"]
            
            return {
                "total_queries": self.metrics["total_queries"],
                "avg_retrieval_time": avg_retrieval,
                "avg_generation_time": avg_generation,
                "avg_total_time": avg_retrieval + avg_generation,
                "cache_hit_rate": self.metrics["cache_hits"] / self.metrics["total_queries"] if self.metrics["total_queries"] > 0 else 0
            }
        
        return self.metrics
    
    def save_state(self, path: str = "./medical_rag_model") -> None:
        """
        Save the RAG pipeline state to disk.
        
        Args:
            path: Directory to save state
        """
        os.makedirs(path, exist_ok=True)
        
        # Save vector store
        self.vector_store.save_index(f"{path}/vector_store")
        
        # Save configuration
        config = {
            "embedding_model_type": self.embedding_generator.model_type,
            "embedding_model_name": self.embedding_generator.model_name,
            "cache_size": self.retriever.cache_size,
            "use_hybrid_search": self.retriever.use_hybrid_search,
            "vector_dimension": self.vector_store.dimension,
            "llm_model": self.llm_processor.model
        }
        
        with open(f"{path}/config.json", "w") as f:
            json.dump(config, f)
        
        logger.info(f"Saved RAG pipeline state to {path}")
    
    @classmethod
    def load_state(cls, path: str = "./medical_rag_model") -> 'RAGPipeline':
        """
        Load the RAG pipeline state from disk.
        
        Args:
            path: Directory to load state from
            
        Returns:
            Initialized RAG pipeline
        """
        # Load configuration
        with open(f"{path}/config.json", "r") as f:
            config = json.load(f)
        
        # Initialize pipeline with saved configuration
        pipeline = cls(
            embedding_model_type=config["embedding_model_type"],
            embedding_model_name=config["embedding_model_name"],
            llm_model=config.get("llm_model", "gpt-4o-mini"),
            use_hybrid_search=config.get("use_hybrid_search", True),
            cache_size=config.get("cache_size", 100)
        )
        
        # Load vector store
        pipeline.vector_store.load_index(f"{path}/vector_store")
        
        logger.info(f"Loaded RAG pipeline state from {path}")
        return pipeline