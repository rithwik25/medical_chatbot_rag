import os
import asyncio
import re
from openai import OpenAI
from typing import List, Tuple, Dict, Any
from data_models import Chunk
from logger import logger

class LLMProcessor:
    """Handles interaction with the LLM API."""
    
    def __init__(self, model: str = "gpt-4o-mini"): # mini for improved latency
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def format_prompt(self, query: str, chunks: List[Tuple[Chunk, float]]) -> str:
        """Format a prompt for the LLM using the query and retrieved chunks."""
        # Construct the context from chunks
        context = "\n\n".join([
            f"Source {i+1} - {chunk.metadata.get('title', 'Unknown')} "
            f"({chunk.metadata.get('url', 'No URL')}) "
            f"Section {chunk.metadata.get('section_index', 'Unknown')}, "
            f"Paragraph {chunk.metadata.get('paragraph_index', 'Unknown')}:\n{chunk.content}"
            for i, (chunk, _) in enumerate(chunks)
        ])
        
        prompt = f"""You are a medical information assistant. Answer the following question based ONLY on the provided sources.
            Your answer should be factual, precise, and well-structured.

            QUESTION: {query}

            SOURCES:
            {context}

            INSTRUCTIONS:
            1. Answer the question using ONLY information from the provided sources.
            2. If the sources don't contain enough information to answer the question completely, say so clearly.
            3. Format your answer in a clear, concise manner.
            4. DO NOT include references or citations in your answer. References will be added separately.
            5. Suggest 2-3 relevant follow-up questions the user might want to ask.
            6. Include a medical disclaimer at the end.

            Your response should follow this structure:
            ANSWER: [Your detailed response without citations]

            FOLLOW-UP QUESTIONS:
            [2-3 suggested follow-up questions]

            DISCLAIMER:
            [Medical disclaimer]
        """
        return prompt
    
    async def generate_response_async(self, query: str, chunks: List[Tuple[Chunk, float]]) -> Dict[str, Any]: # Async function allows handling multiple queries efficiently.
        """Generate a response using the LLM asynchronously."""
        prompt = self.format_prompt(query, chunks)
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.2,
            )
            generated_text = response.choices[0].message.content
            
            # Generate references from chunks before parsing the response
            references = self.generate_references_from_chunks([chunk for chunk, _ in chunks])
            
            result = self.parse_llm_response(query, generated_text)
            result["references"] = references
            
            return result
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return {
                "answer": f"I'm sorry, but I encountered an error while processing your question. Please try again later.",
                "references": [],
                "followup_questions": [],
                "disclaimer": "This information is provided for educational purposes only and is not a substitute for professional medical advice."
            }
    
    def generate_response(self, query: str, chunks: List[Tuple[Chunk, float]]) -> Dict[str, Any]: # Useful when real-time responses are needed.
        """Generate a response using the LLM (synchronous version)."""
        prompt = self.format_prompt(query, chunks)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.2,
            )
            generated_text = response.choices[0].message.content
            
            # Generate references from chunks before parsing the response
            references = self.generate_references_from_chunks([chunk for chunk, _ in chunks])
            
            result = self.parse_llm_response(query, generated_text)
            result["references"] = references
            
            return result
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return {
                "answer": f"I'm sorry, but I encountered an error while processing your question. Please try again later.",
                "references": [],
                "followup_questions": [],
                "disclaimer": "This information is provided for educational purposes only and is not a substitute for professional medical advice."
            }
    
    def generate_references_from_chunks(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """Generate references directly from chunk metadata."""
        references = []
        seen_references = set()  # To track unique references
        
        # Limit to top 3 chunks
        for chunk in chunks[:3]:
            url = chunk.metadata.get("url", "No URL")
            section_index = chunk.metadata.get("section_index", "Unknown")
            paragraph_index = chunk.metadata.get("paragraph_index", "Unknown")
            
            # Create a unique key for this reference to avoid duplicates
            ref_key = f"{url}_{section_index}_{paragraph_index}"
            
            # Skip if we've already seen this exact reference
            if ref_key in seen_references:
                continue
            
            seen_references.add(ref_key)
            
            references.append({
                "url": url,
                "section_index": section_index,
                "paragraph_index": paragraph_index
            })
        
        return references
    
    def parse_llm_response(self, query: str, text: str) -> Dict[str, Any]:
        """Parse the structured response from the LLM."""
        # Initializing response sections
        answer = ""
        followup_questions = []
        disclaimer = ""
        
        # Split by sections
        try:
            # Extract ANSWER section from the response
            if "ANSWER:" in text:
                if "FOLLOW-UP QUESTIONS:" in text:
                    answer_section = text.split("ANSWER:")[1].split("FOLLOW-UP QUESTIONS:")[0].strip()
                else:
                    answer_section = text.split("ANSWER:")[1].strip()
                answer = answer_section
            else:
                # If no sections found, use the whole text as answer
                answer = text
            
            # Extract FOLLOW-UP QUESTIONS section
            if "FOLLOW-UP QUESTIONS:" in text:
                if "DISCLAIMER:" in text:
                    questions_section = text.split("FOLLOW-UP QUESTIONS:")[1].split("DISCLAIMER:")[0].strip()
                else:
                    questions_section = text.split("FOLLOW-UP QUESTIONS:")[1].strip()
                
                question_lines = questions_section.strip().split("\n")
                
                for line in question_lines:
                    line = line.strip()
                    if line and (line.startswith("1.") or line.startswith("2.") or 
                                line.startswith("3.") or line.startswith("-") or
                                line.startswith("•") or re.match(r'^\d+[\).]', line)):
                        # Clean up the question formatting
                        question = re.sub(r'^\d+[\).\s]+|\-\s+|•\s+', '', line).strip()
                        followup_questions.append(question)
            
            # Extract DISCLAIMER section
            if "DISCLAIMER:" in text:
                disclaimer = text.split("DISCLAIMER:")[1].strip()
            else:
                disclaimer = (
                    "This information is provided for educational purposes only and is not a substitute "
                    "for professional medical advice, diagnosis, or treatment. Always seek the advice of "
                    "your physician or other qualified health provider with any questions you may have "
                    "regarding a medical condition."
                )
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            # Fallback to the raw response if parsing fails
            answer = text
            disclaimer = "This information is not medical advice. Consult with healthcare professionals for medical decisions."
        
        # If no follow-up questions were found, generate generic ones
        if not followup_questions:
            # Clean query for follow-up generation
            clean_query = query.lower().replace('?', '').strip()
            topic = re.sub(r'what is|how to|tell me about', '', clean_query).strip()
            
            followup_questions = [
                f"What are the treatment options for {topic}?",
                f"Are there any recent research developments related to {topic}?",
                f"What preventive measures are recommended for {topic}?"
            ]
        
        return {
            "answer": answer,
            "followup_questions": followup_questions[:3],  # Limit to 3
            "disclaimer": disclaimer
        }
    
    def format_final_response(self, result: Dict[str, Any]) -> str:
        """Format the final response with the required sections in specified order."""
        answer = result.get("answer", "")
        references_data = result.get("references", [])
        followup_questions = result.get("followup_questions", [])
        disclaimer = result.get("disclaimer", "")
        
        # Format references section
        references_text = "REFERENCES:\n"
        for i, ref in enumerate(references_data, 1):
            references_text += f"[{i}] {ref['url']}, Section {ref['section_index']}, Paragraph {ref['paragraph_index']}\n"
        
        # Format follow-up questions
        followup_text = "FOLLOW-UP QUESTIONS:\n"
        for i, question in enumerate(followup_questions, 1):
            followup_text += f"{i}. {question}\n"
        
        # Combine all sections in the specified order
        final_response = f"""ANSWER:
            {answer}

            {references_text}
            {followup_text}
            DISCLAIMER:
            {disclaimer}
        """
        return final_response