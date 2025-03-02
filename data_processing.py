import tiktoken
import uuid
import re
from typing import List
from data_models import Document, Chunk
from logger import logger

class HierarchicalDataIngestion:
    """Handles document collection, preprocessing and hierarchical chunking."""
    def __init__(self, max_chunk_size: int = 500):
        self.max_chunk_size = max_chunk_size
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

    def create_document_from_file(self, file_path: str, title: str, publication_date: str = None, url: str = None, source: str = None) -> Document:
        """
        Create a Document object with custom metadata from a text file.
        
        Args:
            file_path: Path to the text file
            title: Title of the document
            publication_date: Publication date (any format)
            url: URL or reference link
            source: Source name (e.g., WHO, CDC)
            
        Returns:
            Document object with content from file and custom metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            document = Document(
                id=str(uuid.uuid4()),
                title=title,
                content=content,
                publication_date=publication_date,
                url=url,
                source=source
            )
            
            logger.info(f"Created document: {title}")
            return document
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return None

    def load_sample_medical_documents(self) -> List[Document]:
        """Load sample medical documents - implementation stays the same as before"""
        # Original implementation remains unchanged
        documents = []
        
        # Document 1
        doc1 = self.create_document_from_file(
            file_path="who_guidelines\dengue.txt",
            title="Dengue and severe dengue",
            publication_date="23rd April 2024",
            url="https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue",
            source="World Health Organization"
        )
        if doc1:
            documents.append(doc1)
        
        # Document 2
        doc2 = self.create_document_from_file(
            file_path="who_guidelines\endometriosis.txt",
            title="Endometriosis",
            publication_date="24th March 2023",
            url="https://www.who.int/news-room/fact-sheets/detail/endometriosis",
            source="World Health Organization"
        )
        if doc2:
            documents.append(doc2)
        
        # Document 3
        doc3 = self.create_document_from_file(
            file_path="who_guidelines\excessive_heat.txt",
            title="Heat and health",
            publication_date="28th May 2024",
            url="https://www.who.int/news-room/fact-sheets/detail/climate-change-heat-and-health",
            source="World Health Organization"
        )
        if doc3:
            documents.append(doc3)
        
        # Document 4
        doc4 = self.create_document_from_file(
            file_path="who_guidelines\herpes.txt",
            title="Herpes simplex virus",
            publication_date="11th December 2024",
            url="https://www.who.int/news-room/fact-sheets/detail/herpes-simplex-virus",
            source="World Health Organization"
        )
        if doc4:
            documents.append(doc4)
        
        # Document 5
        doc5 = self.create_document_from_file(
            file_path="who_guidelines\mental_disorders.txt",
            title="Mental disorders",
            publication_date="8th June 2022",
            url="https://www.who.int/news-room/fact-sheets/detail/mental-disorders",
            source="World Health Organization"
        )
        if doc5:
            documents.append(doc5)
        
        # Document 6
        doc6 = self.create_document_from_file(
            file_path="who_guidelines\mpox.txt",
            title="Mpox",
            publication_date="26th August 2024",
            url="https://www.who.int/news-room/fact-sheets/detail/mpox",
            source="World Health Organization"
        )
        if doc6:
            documents.append(doc6)
        
        logger.info(f"Loaded {len(documents)} sample medical documents")
        return documents

    def _split_into_sections(self, text: str) -> List[dict]:
        """
        Split document into sections based on section headers.
        Uses the pattern from section_paragraph_chunking function.
        
        Args:
            text: The document text
            
        Returns:
            List of dictionaries containing section title and content
        """
        # Pattern to identify section headers (e.g., "Key facts:", "Overview:")
        section_pattern = r'(?:^|\n)([\w\s]+):(?=\s*\n)'
        
        # Find all section headers and their positions
        section_matches = list(re.finditer(section_pattern, text))
        
        sections = []
        
        # If no sections found, treat entire document as one section
        if not section_matches:
            sections.append({
                "title": "Document",
                "content": text
            })
            return sections
        
        # Processing each section
        for i, match in enumerate(section_matches):
            section_title = match.group(1).strip()
            section_start = match.start()
            
            # Determine where this section ends (at the next section or end of document)
            if i < len(section_matches) - 1:
                section_end = section_matches[i + 1].start()
            else:
                section_end = len(text)
            
            # Extracting section content, excluding the header itself
            section_content = text[section_start:section_end].strip()
            section_content = re.sub(r'^[\w\s]+:\s*\n', '', section_content)
            
            sections.append({
                "title": section_title,
                "content": section_content
            })
        
        return sections

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs based on double newlines.
        
        Args:
            text: Section content
            
        Returns:
            List of paragraph strings
        """
        # Splitting on double newlines or multiple newlines
        paragraphs = re.split(r'\n\n+', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Split documents into hierarchical chunks: document → sections → paragraphs.
        Focus only on sections and paragraphs without sentence-level splitting.
        """
        chunks = []
        
        for doc in documents:
            # First level: Splitting document into sections
            sections = self._split_into_sections(doc.content)
            
            for section_idx, section in enumerate(sections):
                section_title = section["title"]
                section_content = section["content"]
                
                # Second level: Splitting sections into paragraphs
                paragraphs = self._split_into_paragraphs(section_content)
                
                for para_idx, paragraph in enumerate(paragraphs):
                    if paragraph.strip():  # Skips empty paragraphs
                        chunk_id = str(uuid.uuid4())
                        chunks.append(Chunk(
                            id=chunk_id,
                            document_id=doc.id,
                            content=paragraph,
                            metadata={
                                "title": doc.title,
                                "publication_date": doc.publication_date,
                                "url": doc.url,
                                "source": doc.source,
                                "section": section_title,
                                "section_index": section_idx+1,
                                "paragraph_index": para_idx+1,
                                "section_total": len(sections),
                                "paragraph_total": len(paragraphs),
                                "hierarchical_id": f"{doc.id}-S{section_idx+1}-P{para_idx+1}"
                            }
                        ))
        
        logger.info(f"Created {len(chunks)} section-paragraph chunks from {len(documents)} documents")
        return chunks