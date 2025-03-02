from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class Document:
    """Represents a medical document with metadata."""
    id: str
    title: str
    content: str
    publication_date: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None # WHO, CDC etc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "publication_date": self.publication_date,
            "url": self.url,
            "source": self.source,
        }

@dataclass
class Chunk:
    """Represents a chunk of text from a document."""
    id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "content": self.content,
            "metadata": self.metadata,
        }

@dataclass
class RAGResponse:
    """The final response from the RAG system."""
    answer: str
    references: List[Dict[str, Any]]
    followup_questions: List[str]
    disclaimer: str
    formatted_response: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "references": self.references,
            "followup_questions": self.followup_questions,
            "disclaimer": self.disclaimer,
            "formatted_response": self.formatted_response,
        }