import faiss
import numpy as np
import os
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi
from data_models import Chunk
from logger import logger

try:
    nltk.data.find('tokenizers/punkt') # checks if punkt(used for tokenization) is installed in nltk
    nltk.data.find('corpora/stopwords') # checks if stopwords(used for removing common words) is installed in nltk
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class HybridVectorStore: # This class implements a Hybrid Vector Store, which combines dense vector search (using FAISS) and sparse text search (using BM25)
    """
    FAISS-based hybrid vector store that combines dense embeddings,
    sparse BM25 retrieval, and metadata filtering.
    """
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        # Dense retrieval with FAISS
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance for dense vectors
        self.chunk_ids = []  # To track IDs of stored text chunks
        self.chunks_map = {}  # Map chunk IDs to their content and metadata
        
        # Sparse retrieval with BM25
        self.bm25 = None
        self.tokenized_corpus = []
        self.corpus_chunk_ids = []
        
        # Weights for hybrid ranking
        self.dense_weight = 0.6
        self.sparse_weight = 0.4
        
        # Stopwords for text processing
        self.stop_words = set(stopwords.words('english')) # loads common words like 'the', 'is', 'and' so they can be removed from search queries
        
        logger.info("Initialized hybrid vector store with dense and sparse retrieval capabilities")
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for BM25 indexing by tokenizing,
        removing stopwords and converting to lowercase.
        """
        tokens = word_tokenize(text.lower()) # converts text to lowercase and splits it into words
        tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words] # Remove stopwords and non-alphabetic tokens
        return tokens
    
    def add_embeddings(self, embeddings: np.ndarray, chunk_ids: List[str], chunks: List[Chunk]):
        """Add embeddings to both dense FAISS index and sparse BM25 index."""
        if len(embeddings) == 0:
            logger.warning("No embeddings to add to index")
            return
        
        # Add to dense FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        embeddings = embeddings.astype('float32')  # FAISS requires float32
        self.index.add(embeddings)
        
        # Store chunk IDs in the same order
        self.chunk_ids = chunk_ids
        
        # Store chunks for later retrieval
        for chunk in chunks:
            self.chunks_map[chunk.id] = chunk # mapping chunk IDs to their content and metadata 
            
        # Build corpus for BM25
        self.tokenized_corpus = []
        self.corpus_chunk_ids = []
        
        for chunk_id in chunk_ids:
            chunk = self.chunks_map[chunk_id]
            # Tokenize and add to corpus
            tokenized_text = self.preprocess_text(chunk.content)
            if tokenized_text:  # Only add if we have tokens
                self.tokenized_corpus.append(tokenized_text)
                self.corpus_chunk_ids.append(chunk_id)
        
        # Initialize BM25
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info(f"Added {len(embeddings)} embeddings to vector store")
        logger.info(f"BM25 index built with {len(self.tokenized_corpus)} documents")
    
    def search(self, query_embedding: np.ndarray, query_text: str, k: int = 5, metadata_filters: Dict[str, Any] = None) -> List[Tuple[Chunk, float]]: # applied dense vector search, sparse BM25 search and metadata filtering to find the most relevant chunks for a given query.
        """
        Hybrid search combining dense embeddings, sparse BM25, and metadata filtering.
        
        Args:
            query_embedding: Vector representation of the query
            query_text: Raw text of the query for BM25 search
            k: Number of results to return
            metadata_filters: Optional filters to apply (e.g., {"source": "WHO"})
            
        Returns:
            List of (chunk, score) tuples sorted by relevance
        """
        # Step 1: Dense retrieval with FAISS
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        dense_k = min(k * 3, len(self.chunk_ids))  # Retrieve more candidates for reranking
        dense_distances, dense_indices = self.index.search(query_embedding, dense_k) # faiss index returns distances and indices of similar vectors
        # faiss returns results in a batch format so we use the 0th index only here(for a single query)

        # Map indices to chunk IDs for dense results
        dense_results = {}
        for i, idx in enumerate(dense_indices[0]): # dense_indices is a 2D array where the first dimension represents the query and the second dimension contains the actual indices of vectors that match the query.
            if 0 <= idx < len(self.chunk_ids):
                chunk_id = self.chunk_ids[idx]
                if chunk_id in self.chunks_map:
                    # Normalize distance to similarity score (closer is better)
                    max_dist = max(dense_distances[0]) if len(dense_distances[0]) > 0 else 1
                    min_dist = min(dense_distances[0]) if len(dense_distances[0]) > 0 else 0
                    dist_range = max(max_dist - min_dist, 1e-5)  # Avoid division by zero
                    
                    # Convert distance to similarity score (1 = best, 0 = worst, range is 0-1)
                    similarity = 1 - ((dense_distances[0][i] - min_dist) / dist_range)
                    dense_results[chunk_id] = similarity
        
        # Step 2: Sparse retrieval with BM25
        sparse_results = {}
        if self.bm25 is not None:
            # Tokenize the query for BM25
            tokenized_query = self.preprocess_text(query_text)
            if tokenized_query:
                # Get BM25 scores
                bm25_scores = self.bm25.get_scores(tokenized_query)
                
                # Normalize BM25 scores
                max_score = max(bm25_scores) if len(bm25_scores) > 0 else 1
                if max_score > 0:
                    normalized_scores = [score / max_score for score in bm25_scores]
                else:
                    normalized_scores = bm25_scores
                
                # Map scores to chunk IDs
                for i, score in enumerate(normalized_scores):
                    if i < len(self.corpus_chunk_ids):
                        chunk_id = self.corpus_chunk_ids[i]
                        sparse_results[chunk_id] = score
        
        # Step 3: Combine results
        combined_results = {}
        all_chunk_ids = set(dense_results.keys()) | set(sparse_results.keys())
        
        for chunk_id in all_chunk_ids:
            dense_score = dense_results.get(chunk_id, 0)
            sparse_score = sparse_results.get(chunk_id, 0)
            
            # Weighted combination
            combined_score = (dense_score * self.dense_weight) + (sparse_score * self.sparse_weight)
            combined_results[chunk_id] = combined_score
        
        # Step 4: Apply metadata filters if provided(for now i am passing none in metadata filters but can be used in various cases)
        filtered_results = {}
        for chunk_id, score in combined_results.items():
            chunk = self.chunks_map.get(chunk_id)
            if chunk is None:
                continue
                
            # Checking if chunk passes all metadata filters
            if metadata_filters:
                passes_filters = True
                for key, value in metadata_filters.items():
                    # Handle nested metadata fields
                    if '.' in key:
                        parts = key.split('.')
                        curr = chunk.metadata
                        for part in parts[:-1]:  # Navigate to the nested field
                            curr = curr.get(part, {})
                        # Check the final field
                        if curr.get(parts[-1]) != value:
                            passes_filters = False
                            break
                    # Direct metadata field
                    elif key not in chunk.metadata or chunk.metadata[key] != value:
                        passes_filters = False
                        break
                
                if not passes_filters:
                    continue
            
            filtered_results[chunk_id] = score
        
        # Sort by score and get top k
        sorted_results = sorted(
            [(self.chunks_map[chunk_id], score) for chunk_id, score in filtered_results.items()],
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        logger.info(f"Hybrid search returned {len(sorted_results)} results")
        return sorted_results

    def exact_match_boost(self, query_text: str, results: List[Tuple[Chunk, float]], boost_factor: float = 1.2) -> List[Tuple[Chunk, float]]: # increases the score of search results if they contain exact phrases from the search query.(bigrams and trigrams)
        """
        Boost scores for chunks containing exact phrases from the query.
        
        Args:
            query_text: The original query
            results: List of (chunk, score) tuples
            boost_factor: Factor to boost scores by for exact matches
            
        Returns:
            Reranked results with boosted scores for exact matches
        """
        # Extract phrases (2+ words) from the query
        words = query_text.lower().split()
        phrases = []
        for i in range(len(words) - 1):
            phrases.append(' '.join(words[i:i+2]))  # Bigrams
        
        # If query has 3+ words, also consider trigrams
        if len(words) >= 3:
            for i in range(len(words) - 2):
                phrases.append(' '.join(words[i:i+3]))  # Trigrams
        
        # Add the full query as a phrase
        if len(words) >= 2:
            phrases.append(query_text.lower())
        
        # Boost scores for results containing exact phrases
        boosted_results = []
        for chunk, score in results:
            chunk_text = chunk.content.lower()
            
            # Check for exact phrase matches
            boost = 1.0
            for phrase in phrases:
                if len(phrase.split()) >= 2 and phrase in chunk_text:  # Only boost for 2+ word phrases
                    boost = boost_factor
                    break
            
            boosted_results.append((chunk, score * boost))
        
        # Re-sort by boosted score
        boosted_results.sort(key=lambda x: x[1], reverse=True)
        return boosted_results
    
    def save_index(self, path: str = "hybrid_index"): # saves the search index (both FAISS vector index and BM25 text data) to disk so it can be reloaded later.
        """Save the hybrid index and metadata to disk."""
        os.makedirs(path, exist_ok=True)
        
        faiss.write_index(self.index, f"{path}/faiss_index.bin") # Saves FAISS index
        
        with open(f"{path}/chunk_ids.json", "w") as f: # Saves chunk IDs
            json.dump(self.chunk_ids, f)
        
        chunks_dict = { # Saves chunks map
            chunk_id: chunk.to_dict() 
            for chunk_id, chunk in self.chunks_map.items()
        }
        with open(f"{path}/chunks.json", "w") as f:
            json.dump(chunks_dict, f)
        
        with open(f"{path}/bm25_corpus.json", "w") as f: # Saves BM25 corpus and related data
            json.dump({
                "tokenized_corpus": self.tokenized_corpus,
                "corpus_chunk_ids": self.corpus_chunk_ids
            }, f)
        
        logger.info(f"Saved hybrid vector store to {path}")
    
    def load_index(self, path: str = "hybrid_index"):
        """Load the hybrid index and metadata from disk."""
        
        self.index = faiss.read_index(f"{path}/faiss_index.bin") # Loads FAISS index
        
        with open(f"{path}/chunk_ids.json", "r") as f: # Loads chunk IDs
            self.chunk_ids = json.load(f)
        
        with open(f"{path}/chunks.json", "r") as f: # Loads chunks map
            chunks_dict = json.load(f)
            
        self.chunks_map = {} # Converts back to Chunk objects
        for chunk_id, chunk_dict in chunks_dict.items():
            self.chunks_map[chunk_id] = Chunk(
                id=chunk_dict["id"],
                document_id=chunk_dict["document_id"],
                content=chunk_dict["content"],
                metadata=chunk_dict["metadata"]
            )
        
        with open(f"{path}/bm25_corpus.json", "r") as f: # Loads BM25 corpus and rebuilds BM25 index
            bm25_data = json.load(f)
            self.tokenized_corpus = bm25_data["tokenized_corpus"]
            self.corpus_chunk_ids = bm25_data["corpus_chunk_ids"]
        
        if self.tokenized_corpus: # Reinitializes BM25
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info(f"Loaded hybrid vector store from {path} with {len(self.chunk_ids)} chunks")
    
    def set_weights(self, dense_weight: float = 0.6, sparse_weight: float = 0.4):
        """
        Set the weights for hybrid search combination.
        
        Args:
            dense_weight: Weight for dense vector search (0-1)
            sparse_weight: Weight for sparse BM25 search (0-1)
        """
        total = dense_weight + sparse_weight # total = 1
        self.dense_weight = dense_weight / total
        self.sparse_weight = sparse_weight / total
        logger.info(f"Set hybrid weights: dense={self.dense_weight:.2f}, sparse={self.sparse_weight:.2f}")
