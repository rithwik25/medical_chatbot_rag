import os
import numpy as np
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from logger import logger
from typing import List, Tuple
from data_models import Chunk

class EmbeddingGenerator: 
    """Handles text-to-embedding conversion using various embedding models."""
    
    def __init__(self, model_type: str = "openai", model_name: str = "text-embedding-3-small"):
        """
        Initialize the embedding generator with the specified model.
        
        Args:
            model_type: Type of embedding model to use ('openai', 'huggingface', or 'medical')
            model_name: Name of the specific model to use
        """
        self.model_type = model_type
        self.model_name = model_name
        
        if model_type == "openai":
            # Initialize OpenAI client
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info(f"Using OpenAI embedding model: {model_name}")
            
        elif model_type == "huggingface":
            # Initialize HuggingFace model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name) # tokenizer converts text into tokens
            self.model = AutoModel.from_pretrained(model_name) # model converts tokens into embeddings(sentence-transformers/all-mpnet-base-v2)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info(f"Using HuggingFace embedding model: {model_name} on {self.device}")
            
        elif model_type == "medical":
            # Initialize medical domain-specific model
            # Examples: BioBERT, ClinicalBERT, or BlueBERT
            if model_name == "BiomedBERT":
                self.model_name = "emilyalsentzer/Bio_ClinicalBERT"  # A good medical model
            elif model_name == "BioBERT":
                self.model_name = "dmis-lab/biobert-base-cased-v1.1"
            elif model_name == "SciBERT":
                self.model_name = "allenai/scibert_scivocab_uncased"
            else:
                self.model_name = model_name
                
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info(f"Using medical embedding model: {self.model_name} on {self.device}")
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embeddings for a single text."""
        if self.model_type == "openai":
            # Use OpenAI's embedding API
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model_name
                )
                embedding = np.array(response.data[0].embedding)
                # Normalize the embedding
                return embedding / np.linalg.norm(embedding)
            except Exception as e:
                logger.error(f"Error with OpenAI embedding API: {str(e)}")
                # Return a zero vector as fallback
                return np.zeros(1536)  # OpenAI's text-embedding-3-small has 1536 dimensions
            
        else:  # HuggingFace or medical models
            # Tokenize and encode
            inputs = self.tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.device)

            # Generating embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Using mean pooling of token embeddings as the sentence embedding
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            
            # Mask padding tokens
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            # Sum the masked embeddings and divide by the sum of the mask
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            
            # Convert to numpy array and normalize
            embedding_np = embedding.cpu().numpy()[0]
            embedding_norm = embedding_np / np.linalg.norm(embedding_np)
            
            return embedding_norm
    
    def generate_embeddings(self, chunks: List[Chunk]) -> Tuple[np.ndarray, List[str]]:
        """Generate embeddings for a list of chunks."""
        embeddings = []
        chunk_ids = []
        
        for i, chunk in enumerate(chunks):
            try:
                embedding = self.generate_embedding(chunk.content)
                embeddings.append(embedding)
                chunk_ids.append(chunk.id)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Generated embeddings for {i + 1}/{len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error generating embedding for chunk {chunk.id}: {str(e)}")
        
        logger.info(f"Completed generating {len(embeddings)} embeddings")
        return np.array(embeddings), chunk_ids