import time
import asyncio
import numpy as np
from typing import List, Tuple, Dict, Any
from logger import logger
from data_models import Chunk
from embedding_generator import EmbeddingGenerator
from vector_store import HybridVectorStore

class Retriever:
    """Handles retrieving relevant chunks for a query with optimized latency."""
    
    def __init__(self, embedding_generator: EmbeddingGenerator, vector_store: HybridVectorStore, use_hybrid_search: bool = True, pre_filter_metadata: dict = None, cache_size: int = 100):
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.use_hybrid_search = use_hybrid_search
        self.pre_filter_metadata = pre_filter_metadata
        
        # Add embedding cache for frequent queries
        self.embedding_cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.total_queries = 0
    
    async def retrieve_async(self, query: str, k: int = 5, filter_params: dict = None) -> List[Tuple[Chunk, float]]: # retrieves k most relevant text chunks for a query asynchronously.
        """Asynchronously retrieve the top-k most relevant chunks for a query."""
        start_time = time.time()
        self.total_queries += 1
        
        # Generates embedding asynchronously with caching
        query_key = query.lower().strip()
        if query_key in self.embedding_cache: # If the query exists in the cache, retrieves its embedding and increases cache_hits.
            query_embedding = self.embedding_cache[query_key]
            self.cache_hits += 1
            logger.debug(f"Cache hit for query: {query_key} (hit rate: {self.cache_hits/self.total_queries:.2f})")
        else:
            query_embedding = await self._generate_embedding_async(query)
            # Updating cache with LRU policy
            if len(self.embedding_cache) >= self.cache_size:
                # Removing oldest item
                self.embedding_cache.pop(next(iter(self.embedding_cache)))
            self.embedding_cache[query_key] = query_embedding
        
        # Applying filters
        effective_filters = filter_params or self._get_filter_params(query)
        
        # Searching with hybrid approach if enabled
        try:
            if self.use_hybrid_search:
                results = await self._hybrid_search_async(query, query_embedding, k, effective_filters)
            else:
                results = await self._vector_search_async(query_embedding, query, k, effective_filters)
            
            # Track and log performance metrics
            elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            logger.info(f"Retrieved {len(results)} chunks in {elapsed_time:.2f}ms: {query[:50]}...")

            # Validates that hierarchical indices are included in all results
            for chunk, _ in results:
                if "section_index" not in chunk.metadata or "paragraph_index" not in chunk.metadata:
                    logger.warning(f"Missing hierarchical indices in chunk {chunk.id}")

            # Alert if retrieval is slow
            if elapsed_time > 500:
                logger.warning(f"Retrieval latency ({elapsed_time:.2f}ms) exceeds threshold")
                
            return results
            
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            # Return empty results on error
            return []
    
    async def _generate_embedding_async(self, query: str):
        """Generate embedding asynchronously with retry logic."""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                return await asyncio.to_thread(self.embedding_generator.generate_embedding, query) # calls generate_embedding() in a background thread
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Embedding generation failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Embedding generation failed after {max_retries} attempts: {str(e)}")
                    # Return a zero vector as fallback
                    return np.zeros(self.vector_store.dimension)
    
    async def _vector_search_async(self, query_embedding, query_text: str, k: int, filter_params: dict = None):
        """Perform vector search asynchronously."""
        return await asyncio.to_thread( # calls vector_store.search() in a separate thread
            self.vector_store.search, 
            query_embedding,
            query_text,
            k=k, 
            metadata_filters=filter_params
        )
    
    async def _hybrid_search_async(self, query: str, query_embedding, k: int, filter_params: dict = None):
        """Perform hybrid search with exact match boosting."""
        # Get initial hybrid search results
        results = await self._vector_search_async(query_embedding, query, k=k*2, filter_params=filter_params)
        
        boosted_results = await asyncio.to_thread( # Boosts results with exact match phrases
            self.vector_store.exact_match_boost,
            query,
            results
        )
        return boosted_results[:k] # Returns top k after boosting
    
    def _get_filter_params(self, query: str) -> dict:
        """Generate intelligent filter parameters based on query content."""
        filter_params = {}
        
        # Base filters from initialization
        if self.pre_filter_metadata:
            filter_params.update(self.pre_filter_metadata)
        
        # Enhanced query understanding
        query_lower = query.lower()
        
        # Detects temporal filters
        #if any(term in query_lower for term in ["recent", "latest", "new", "current"]):
        #    filter_params["recency"] = {"min_date": "2023-01-01"}
        #elif any(term in query_lower for term in ["2024", "this year"]):
        #    filter_params["recency"] = {"min_date": "2024-01-01"}
            
        # Detects source preferences
        if "who" in query_lower or "world health" in query_lower:
            filter_params["source"] = "World Health Organization"
        elif "cdc" in query_lower or "disease control" in query_lower:
            filter_params["source"] = "Centers for Disease Control and Prevention"
        return filter_params
    
    def analyze_query(self, query: str) -> Dict[str, Any]: # Extracts entities, topics, and classifies query types (e.g., "treatment", "symptoms", "causes").
        """
        Analyze query to extract entities, topics and semantic type.
        Used to improve retrieval understanding.
        """
        query_info = {
            "original_query": query,
            "entities": [],
            "query_type": "informational",  # default
            "medical_topics": []
        }
        
        # Simple entity extraction (could use NER in production)
        medical_conditions = ["dengue", "mpox", "endometriosis", "heat", "herpes", "disorder"]
        for condition in medical_conditions:
            if condition in query.lower():
                query_info["entities"].append({"type": "medical_condition", "value": condition})
                query_info["medical_topics"].append(condition)
        
        # Query type classification
        if any(q in query.lower() for q in ["how to", "treatment", "manage"]):
            query_info["query_type"] = "treatment"
        elif any(q in query.lower() for q in ["symptom", "sign", "feel"]):
            query_info["query_type"] = "symptoms"
        elif any(q in query.lower() for q in ["cause", "risk", "factor"]):
            query_info["query_type"] = "causes"

        return query_info
        
    # Synchronous version for backward compatibility(required for flask)
    def retrieve(self, query: str, k: int = 5, filter_params: dict = None) -> List[Tuple[Chunk, float]]: # float is for the relevance score
        """Retrieve the top-k most relevant chunks for a query (sync version)."""
        return asyncio.run(self.retrieve_async(query, k, filter_params))