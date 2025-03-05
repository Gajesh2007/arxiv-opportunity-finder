"""
Claude client for analyzing papers.
Uses Claude 3.7 Sonnet to directly analyze PDF documents.
"""
import os
import sys
import json
import logging
import asyncio
import base64
import tempfile
import io
import time
import random
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from anthropic import AsyncAnthropic, Anthropic, BadRequestError, RateLimitError
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.config import config

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class ClaudeClient:
    """
    Client for interacting with Claude API to analyze papers.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the Claude client.
        
        Args:
            api_key: Optional API key for Claude.
                   If not provided, uses the ANTHROPIC_API_KEY environment variable.
            model: Optional model to use.
                  If not provided, uses the CLAUDE_MODEL environment variable
                  or falls back to claude-3-7-sonnet-latest.
        """
        # Get API key from parameter, environment variable, or config
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY") or config.get("ANTHROPIC_API_KEY")
            
        if api_key is None:
            raise ValueError("Claude API key is required.")
        
        # Get model from parameter, environment variable, or config
        if model is None:
            model = os.getenv("CLAUDE_MODEL") or config.get("CLAUDE_MODEL") or "claude-3-7-sonnet-latest"
        
        self.model = model
        
        # Initialize the Claude client
        self.client = AsyncAnthropic(api_key=api_key)
        
        logger.info(f"Claude client initialized with model: {self.model}")
    
    async def analyze_paper(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyze a paper using Claude 3.7 Sonnet directly with the PDF.
        If the PDF is too large, falls back to various methods:
        1. PDF chunking with descriptive conversion
        2. Text extraction
        3. Text chunking
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing the analysis results
        """
        logger.info(f"Analyzing paper: {pdf_path}")
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise ValueError(f"PDF file not found: {pdf_path}")
        
        # Prepare the system prompt
        system_prompt = """You are a research analyst tasked with evaluating the innovation potential and real-world applicability of research papers. 
Your goal is to identify papers that present novel ideas with high potential for proof-of-concept implementation.

Be truly critical in your assessment - scores should vary significantly between different papers. Be honest when a paper has limited innovation potential or significant implementation challenges. Not every paper deserves high scores.

When providing scores, use precise decimal values (e.g., 7.3, 8.6, 5.2) rather than just integers to better capture nuanced differences between papers. This precision helps distinguish between papers with similar qualities.

Analyze the provided research paper and evaluate it on the following criteria:
1. Innovation Score (0.0-10.0): How novel and groundbreaking is the research? Does it present truly new ideas or approaches? Use decimal precision.
2. PoC Potential Score (0.0-10.0): How feasible would it be to create a meaningful proof-of-concept implementation of the paper's key ideas? Use decimal precision.
3. Wow Factor Score (0.0-10.0): How exciting or impressive are the results? Would the implementation impress people? Use decimal precision.
4. Implementation Complexity Score (0.0-10.0, lower is better): How complex would implementing the core ideas be? Lower scores indicate simpler implementation. Use decimal precision.

For each paper, extract the following information:
- Paper Title
- Authors
- Key Research Questions
- Novel Methodology/Approach
- Main Findings/Results
- Technical Implementation Details
- Required Resources for Implementation
- Potential Applications
- Implementation Challenges

Your analysis should be detailed and specific, focusing on the technical aspects and practical implementation potential.
"""
        
        # Prepare the user prompt
        user_prompt = "Please analyze this research paper and provide your evaluation based on the criteria."
        
        try:
            # First try the direct PDF approach
            try:
                # Read the PDF file and encode as base64
                with open(pdf_path, "rb") as f:
                    pdf_content = base64.b64encode(f.read()).decode("utf-8")
                
                # Create a message with the PDF attachment using newer document format
                response = await self.client.messages.create(
                    model=self.model,
                    system=system_prompt,
                    max_tokens=64000,
                    thinking={"type": "enabled", "budget_tokens": 63000},
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_prompt},
                                {
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "application/pdf",
                                        "data": pdf_content
                                    }
                                }
                            ]
                        }
                    ],
                    stream=True
                )
                
                # Extract the content
                content = response.content[0].text
                
            except BadRequestError as e:
                error_message = str(e)
                if "prompt is too long" in error_message or "tokens > " in error_message:
                    logger.warning(f"PDF is too large for direct processing. Trying PDF chunking approach: {e}")
                    try:
                        # First try PDF chunking with descriptive conversion
                        content = await self._analyze_with_pdf_chunking(pdf_path, system_prompt, user_prompt)
                    except Exception as chunk_error:
                        logger.warning(f"PDF chunking approach failed: {chunk_error}. Falling back to text extraction.")
                        content = await self._analyze_with_text_extraction(pdf_path, system_prompt, user_prompt)
                else:
                    # Re-raise if it's not a token limit issue
                    raise
            
            # Process the response
            analysis = self._process_response(content)
            
            logger.info(f"Analysis complete for paper: {pdf_path.name}")
            return analysis
        
        except Exception as e:
            logger.error(f"Failed to analyze paper: {e}", exc_info=True)
            raise
    
    async def _analyze_with_pdf_chunking(self, pdf_path: Path, system_prompt: str, user_prompt: str) -> str:
        """
        Analyze a paper by chunking the PDF and getting descriptive text for each chunk.
        Uses appropriate sized chunks and implements rate limiting with reasonable delays.
        Always uses streaming for API calls to handle long-running operations.
        
        Args:
            pdf_path: Path to the PDF file
            system_prompt: System prompt for the final analysis
            user_prompt: User prompt for the final analysis
            
        Returns:
            Analysis text from Claude
        """
        logger.info(f"Using PDF chunking and descriptive conversion for: {pdf_path}")
        
        # Split the PDF into chunks
        pdf_chunks = self._split_pdf_into_chunks(pdf_path)
        logger.info(f"Split PDF into {len(pdf_chunks)} chunks")
        
        # Process each chunk to get descriptive text - SEQUENTIALLY to avoid rate limits
        descriptive_texts = []
        
        chunk_system_prompt = """You are a PDF interpreter that provides a comprehensive textual representation of research papers.
Your task is to convert everything in this PDF chunk into descriptive text:

1. Preserve all text content exactly as written
2. For images, provide concise descriptions of what they depict (focus on technical content)
3. For diagrams/charts/graphs, describe their key components, labels, axes, data trends, and what they illustrate
4. For mathematical equations, preserve them exactly
5. For tables, describe their structure and content clearly
6. For code blocks, preserve them exactly

Be thorough but efficient in your descriptions. This is part of a larger academic paper, so focus on capturing the scholarly content accurately.
This description will be used for subsequent analysis, so include all meaningful technical details.
"""
        
        # Rate limit handling constants
        RATE_LIMIT_TOKENS_PER_MINUTE = 20000  # Based on the error message
        BASE_DELAY = 15  # Reduced base delay in seconds between chunks (15 seconds)
        MAX_DELAY = 120  # Maximum delay in seconds (2 minutes)
        MAX_RETRIES = 5
        
        for i, chunk_data in enumerate(pdf_chunks):
            chunk_prompt = f"This is chunk {i+1} of {len(pdf_chunks)} from a research paper. Convert this PDF chunk into a comprehensive textual representation that preserves all information:"
            
            # More reasonable token estimation - approx 1KB base64 = 3/4 KB raw = ~200 tokens
            # This is a conservative estimate but better than our previous approach
            estimated_tokens = len(chunk_data) / 20  # Much more conservative estimate
            
            # Calculate delay based on estimated tokens to stay within rate limits
            # Cap the delay at MAX_DELAY seconds
            delay_seconds = min(MAX_DELAY, max(BASE_DELAY, (estimated_tokens / RATE_LIMIT_TOKENS_PER_MINUTE) * 60))
            
            logger.info(f"Processing PDF chunk {i+1}/{len(pdf_chunks)} - estimated {estimated_tokens:.0f} tokens, delay: {delay_seconds:.1f}s")
            
            # If not the first chunk, wait to respect rate limits
            if i > 0:
                logger.info(f"Waiting {delay_seconds:.1f} seconds before processing next chunk to respect rate limits")
                await asyncio.sleep(delay_seconds)
            
            # Try to process this chunk with retries for rate limit errors
            retry_count = 0
            while retry_count < MAX_RETRIES:
                try:
                    # Always use streaming for potentially long-running operations
                    content_pieces = []
                    async with self.client.messages.stream(
                        model=self.model,
                        system=chunk_system_prompt,
                        max_tokens=32000,  # Increasing max tokens for larger chunk outputs
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": chunk_prompt},
                                    {
                                        "type": "document",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "application/pdf",
                                            "data": chunk_data
                                        }
                                    }
                                ]
                            }
                        ]
                    ) as stream:
                        async for message in stream:
                            if message.type == "content_block_delta" and message.delta.type == "text":
                                content_pieces.append(message.delta.text)
                    
                    # Combine the content pieces
                    chunk_text = "".join(content_pieces)
                    descriptive_texts.append(chunk_text)
                    logger.info(f"Processed PDF chunk {i+1}/{len(pdf_chunks)} - generated {len(chunk_text)} characters of descriptive text")
                    break  # Success, exit retry loop
                    
                except RateLimitError as e:
                    retry_count += 1
                    # Exponential backoff with jitter
                    backoff_time = min(120, (2 ** retry_count) * (0.8 + 0.4 * random.random()))
                    logger.warning(f"Rate limit exceeded on chunk {i+1}, attempt {retry_count}/{MAX_RETRIES}. Backing off for {backoff_time:.1f}s: {e}")
                    
                except Exception as e:
                    logger.error(f"Error processing PDF chunk {i+1}: {e}")
                    # If a chunk fails, add a placeholder and continue
                    descriptive_texts.append(f"[Error processing chunk {i+1}: {str(e)}]")
                    break  # Non-rate limit error, continue to next chunk
        
        # If we have no successful descriptive texts, try the text extraction fallback
        if all("[Error" in text for text in descriptive_texts):
            logger.warning("All PDF chunks failed to process. Falling back to text extraction method.")
            return await self._analyze_with_text_extraction(pdf_path, system_prompt, user_prompt)
        
        # For very large papers, we might need to be more selective with the combined text
        if len(pdf_chunks) > 2:
            logger.info("Multiple chunks detected - extracting key sections from each chunk's descriptive text")
            processed_texts = []
            for i, text in enumerate(descriptive_texts):
                # Extract important elements from each chunk's text
                key_text = self._extract_key_elements(text)
                processed_texts.append(f"--- SECTION {i+1} ---\n{key_text}")
            combined_text = "\n\n".join(processed_texts)
        else:
            # For smaller papers, use the full descriptive text
            combined_text = "\n\n".join([
                f"--- SECTION {i+1} ---\n{text}" 
                for i, text in enumerate(descriptive_texts)
            ])
        
        # Use the combined descriptive text for the final analysis
        # Wait a bit to respect rate limits before the final analysis
        await asyncio.sleep(BASE_DELAY)
        
        logger.info(f"Performing final analysis on combined descriptive text ({len(combined_text)} characters)")
        combined_prompt = f"{user_prompt}\n\nBelow is a detailed textual representation of the entire paper (created from PDF chunks):\n\n{combined_text}"
        
        # Retry logic for final analysis
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                # Always use streaming for final analysis
                content_pieces = []
                async with self.client.messages.stream(
                    model=self.model,
                    system=system_prompt,
                    max_tokens=32000,  # Increased for more detailed analysis
                    messages=[
                        {
                            "role": "user", 
                            "content": combined_prompt
                        }
                    ]
                ) as stream:
                    async for message in stream:
                        if message.type == "content_block_delta" and message.delta.type == "text":
                            content_pieces.append(message.delta.text)
                
                # Combine the content pieces
                return "".join(content_pieces)
                
            except BadRequestError as e:
                # If too large for context window, extract key sections
                if "prompt is too long" in str(e) or "tokens > " in str(e):
                    logger.warning(f"Combined descriptive text still too large ({len(combined_text)} chars). Extracting key sections: {e}")
                    # Extract introduction, methodology, results, and conclusion sections
                    extracted_text = self._extract_key_sections(combined_text)
                    
                    # Wait a bit to respect rate limits before retrying with smaller content
                    await asyncio.sleep(BASE_DELAY / 2)
                    
                    truncated_prompt = f"{user_prompt}\n\nBelow are the key sections extracted from the paper:\n\n{extracted_text}"
                    
                    # Retry with the smaller content, using streaming
                    content_pieces = []
                    async with self.client.messages.stream(
                        model=self.model,
                        system=system_prompt,
                        max_tokens=32000,
                        messages=[
                            {
                                "role": "user", 
                                "content": truncated_prompt
                            }
                        ]
                    ) as stream:
                        async for message in stream:
                            if message.type == "content_block_delta" and message.delta.type == "text":
                                content_pieces.append(message.delta.text)
                    
                    # Combine the content pieces
                    return "".join(content_pieces)
                else:
                    raise
                
            except RateLimitError as e:
                retry_count += 1
                # Exponential backoff with jitter
                backoff_time = min(120, (2 ** retry_count) * (0.8 + 0.4 * random.random()))
                logger.warning(f"Rate limit exceeded on final analysis, attempt {retry_count}/{MAX_RETRIES}. Backing off for {backoff_time:.1f}s: {e}")
                
                if retry_count < MAX_RETRIES:
                    await asyncio.sleep(backoff_time)
                else:
                    logger.error(f"Failed to complete final analysis after {MAX_RETRIES} retries")
                    return "ERROR: Rate limits prevented complete analysis. Partial results:\n\n" + "\n\n".join(descriptive_texts[:2])
            
            except Exception as e:
                logger.error(f"Error in final analysis: {e}")
                # Return whatever we've got
                return "ERROR in combined analysis. Partial results:\n\n" + "\n\n".join(descriptive_texts[:2])
    
    def _split_pdf_into_chunks(self, pdf_path: Path) -> List[str]:
        """
        Split a PDF into chunks of appropriate size for Claude.
        Targets approximately 100K tokens per chunk to reduce the number of API calls.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of base64-encoded PDF chunks
        """
        try:
            from pypdf import PdfReader, PdfWriter
            
            # Open the PDF file
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)
            
            # Calculate chunk size (pages per chunk)
            # Use a more conservative estimation: ~1000 tokens per page for academic papers
            # This is a safer estimate than our previous approach
            estimated_tokens_per_page = 1000
            
            # Target ~80K tokens per chunk (well below Claude's limit)
            target_tokens_per_chunk = 80000
            pages_per_chunk = max(1, int(target_tokens_per_chunk / estimated_tokens_per_page))
            
            logger.info(f"PDF has {total_pages} pages, targeting {pages_per_chunk} pages per chunk (~{estimated_tokens_per_page} tokens/page)")
            
            # Ensure we don't create too many chunks but also keep chunks reasonably sized
            max_chunks = 4
            max_pages_per_chunk = 25  # Set a hard limit on pages per chunk
            
            if total_pages > 0 and (total_pages / pages_per_chunk) > max_chunks:
                pages_per_chunk = max(1, min(max_pages_per_chunk, total_pages // max_chunks))
                logger.info(f"Adjusted to {pages_per_chunk} pages per chunk to limit total chunks to {max_chunks}")
            
            # Make sure we don't exceed our maximum pages per chunk
            pages_per_chunk = min(pages_per_chunk, max_pages_per_chunk)
            
            # Create chunks
            chunks = []
            for i in range(0, total_pages, pages_per_chunk):
                # Create a PdfWriter for this chunk
                writer = PdfWriter()
                
                # Add pages to this chunk
                end_page = min(i + pages_per_chunk, total_pages)
                for page_num in range(i, end_page):
                    writer.add_page(reader.pages[page_num])
                
                # Write the chunk to a bytes buffer
                buffer = io.BytesIO()
                writer.write(buffer)
                buffer.seek(0)
                
                # Encode the chunk as base64
                chunk_data = base64.b64encode(buffer.read()).decode('utf-8')
                chunks.append(chunk_data)
            
            logger.info(f"Created {len(chunks)} PDF chunks with approximately {pages_per_chunk} pages per chunk")
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting PDF into chunks: {e}")
            raise
    
    def _extract_key_elements(self, text: str) -> str:
        """
        Extract key elements from a chunk's descriptive text to reduce size.
        
        Args:
            text: Descriptive text from a PDF chunk
            
        Returns:
            Extracted key elements
        """
        # Define patterns to identify key parts of a research paper
        key_patterns = [
            # Look for headings
            (r"(?:^|\n)(?:# |## |### |)[A-Z][A-Za-z0-9 ]+\n", "heading"),
            # Look for abstracts
            (r"(?:^|\n)(?:Abstract|ABSTRACT)(?::|\.|\n)", "abstract"),
            # Look for equations
            (r"(?:\$\$.*?\$\$|\$.*?\$)", "equation"),
            # Look for tables (captures table descriptions)
            (r"(?:Table|TABLE)[^.]*?(?:\.|:)[^.]+\.", "table"),
            # Look for figures (captures figure descriptions)
            (r"(?:Figure|Fig\.|FIG\.)[^.]*?(?:\.|:)[^.]+\.", "figure"),
            # Look for numerical results
            (r"(?:results?|accuracy|precision|recall|f1|score|performance)[^.]*?(?:\d+\.\d+%?|\d+%)[^.]+\.", "result"),
            # Look for methodology descriptions
            (r"(?:method|approach|algorithm|architecture|model|framework)[^.]*?(?:consists of|based on|uses|employs|proposes)[^.]+\.", "method"),
            # Look for conclusions
            (r"(?:^|\n)(?:Conclusion|CONCLUSION|In conclusion|To conclude)(?:s|:|\.|\n)", "conclusion")
        ]
        
        # Extract important parts
        extracted_lines = []
        lines = text.split('\n')
        
        # Always include the first few lines (likely title, authors, abstract)
        intro_lines = min(15, len(lines) // 10)
        extracted_lines.extend(lines[:intro_lines])
        
        # Scan for important patterns
        for i, line in enumerate(lines):
            if i < intro_lines:
                continue  # Already included
            
            for pattern, element_type in key_patterns:
                import re
                if re.search(pattern, line, re.IGNORECASE):
                    # Include this line and a few following lines for context
                    context_lines = min(5, len(lines) - i)
                    extracted_lines.extend(lines[i:i+context_lines])
                    break
        
        # Always include some lines from the end (likely conclusion)
        conclusion_lines = min(15, len(lines) // 10)
        if conclusion_lines > 0:
            extracted_lines.extend(lines[-conclusion_lines:])
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for line in extracted_lines:
            line_key = line.strip()
            if line_key and line_key not in seen:
                seen.add(line_key)
                result.append(line)
        
        return '\n'.join(result)
    
    def _extract_key_sections(self, text: str) -> str:
        """
        Extract key sections from the paper text.
        
        Args:
            text: Combined descriptive text of the paper
            
        Returns:
            Extracted key sections
        """
        # Define key sections to look for
        key_sections = [
            ("abstract", ["abstract", "summary"]),
            ("introduction", ["introduction", "background", "motivation"]),
            ("methodology", ["methodology", "method", "approach", "proposed", "implementation"]),
            ("results", ["results", "evaluation", "experiment", "performance"]),
            ("conclusion", ["conclusion", "discussion", "future work"])
        ]
        
        extracted_sections = []
        remaining_text = text
        
        # Extract each section
        for section_name, keywords in key_sections:
            section_text = None
            
            # Look for section headers
            for keyword in keywords:
                # Find potential section headers
                patterns = [
                    f"\n{keyword.title()}\n",
                    f"\n{keyword.upper()}\n",
                    f"\n{keyword.capitalize()}\n",
                    f"\n{section_name.upper()}\n",
                    f"--- SECTION .* ---\n.*{keyword}",
                ]
                
                for pattern in patterns:
                    import re
                    matches = list(re.finditer(pattern, remaining_text, re.IGNORECASE))
                    if matches:
                        # Found a potential section
                        start_idx = matches[0].start()
                        
                        # Find the next section header or the end
                        next_section_match = None
                        for next_keyword, _ in key_sections:
                            next_matches = list(re.finditer(f"\n{next_keyword}\n", remaining_text[start_idx+1:], re.IGNORECASE))
                            if next_matches:
                                if next_section_match is None or next_matches[0].start() < next_section_match[1]:
                                    next_section_match = (next_keyword, next_matches[0].start() + start_idx + 1)
                        
                        end_idx = next_section_match[1] if next_section_match else len(remaining_text)
                        section_text = remaining_text[start_idx:end_idx].strip()
                        break
                
                if section_text:
                    break
            
            # If section was found, add it
            if section_text:
                extracted_sections.append(f"--- {section_name.upper()} ---\n{section_text}")
        
        # If no sections were extracted, take the beginning, middle, and end
        if not extracted_sections:
            lines = text.split('\n')
            num_lines = len(lines)
            
            # Take first 50 lines
            beginning = '\n'.join(lines[:min(50, num_lines // 3)])
            
            # Take middle 50 lines
            middle_start = max(0, num_lines // 2 - 25)
            middle = '\n'.join(lines[middle_start:middle_start + min(50, num_lines // 3)])
            
            # Take last 50 lines
            end = '\n'.join(lines[max(0, num_lines - min(50, num_lines // 3)):])
            
            extracted_sections = [
                "--- BEGINNING ---\n" + beginning,
                "--- MIDDLE ---\n" + middle,
                "--- END ---\n" + end
            ]
        
        return "\n\n".join(extracted_sections)
    
    async def _analyze_with_text_extraction(self, pdf_path: Path, system_prompt: str, user_prompt: str) -> str:
        """
        Fallback method for analyzing papers by first extracting text from the PDF.
        
        Args:
            pdf_path: Path to the PDF file
            system_prompt: System prompt for Claude
            user_prompt: User prompt for Claude
            
        Returns:
            Analysis text from Claude
        """
        logger.info(f"Using text extraction fallback for: {pdf_path}")
        
        # Extract text from PDF using PyPDF2
        extracted_text = self._extract_text_from_pdf(pdf_path)
        
        # If text extraction fails or extracts too little text, try alternative methods
        if not extracted_text or len(extracted_text) < 1000:
            logger.warning(f"Primary text extraction yielded insufficient results, trying alternative methods")
            extracted_text = self._extract_text_with_pdfplumber(pdf_path)
            
        # If still not enough text, return error message
        if not extracted_text or len(extracted_text) < 1000:
            logger.error(f"All text extraction methods failed for {pdf_path}")
            return "ERROR: Could not extract sufficient text from the PDF to analyze."
            
        try:
            # Add context about the extraction method to the prompt
            fallback_prompt = f"{user_prompt}\n\nNote: Due to size limitations, I'm analyzing the extracted text from the PDF rather than the direct PDF. The formatting and some elements may be lost in extraction."
            
            # Send the extracted text to Claude
            response = await self.client.messages.create(
                model=self.model,
                system=system_prompt,
                max_tokens=64000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{fallback_prompt}\n\n{extracted_text}"}
                        ]
                    }
                ]
            )
            
            return response.content[0].text
            
        except BadRequestError as e:
            # If still too large, try chunking the text
            if "prompt is too long" in str(e) or "tokens > " in str(e):
                logger.warning(f"Extracted text is still too large. Attempting chunking approach: {e}")
                return await self._analyze_with_text_chunking(extracted_text, system_prompt)
            else:
                raise
    
    async def _analyze_with_text_chunking(self, text: str, system_prompt: str) -> str:
        """
        Analyze a paper by chunking its text and analyzing each chunk separately.
        
        Args:
            text: Extracted text from paper
            system_prompt: System prompt for Claude
            
        Returns:
            Combined analysis from Claude
        """
        logger.info("Using text chunking strategy for large document")
        
        # Split text into roughly equal chunks (estimating about 5000 tokens per chunk)
        # A rough approximation is 4 characters per token
        chunk_size = 20000  # characters
        chunks = self._split_text_into_chunks(text, chunk_size)
        
        # If we have too many chunks, prioritize beginning and end of the paper
        max_chunks = 4
        if len(chunks) > max_chunks:
            logger.warning(f"Document is very large with {len(chunks)} chunks. Using first and last chunks.")
            selected_chunks = chunks[:2] + chunks[-2:]  # Take first 2 and last 2 chunks
        else:
            selected_chunks = chunks
            
        chunk_analyses = []
        chunk_system_prompt = """You are analyzing a PORTION of a research paper. 
Provide observations about this section, focusing on any of these aspects that you can identify:
- Research questions
- Methodology/approach
- Results/findings
- Technical implementation details
- Resources needed
- Potential applications
- Implementation challenges

DO NOT try to assign scores yet, as you're only seeing a portion of the paper.
"""
        
        # Analyze each chunk
        for i, chunk in enumerate(selected_chunks):
            chunk_prompt = f"This is chunk {i+1} of {len(selected_chunks)} from a research paper. Please analyze this section:\n\n{chunk}"
            
            try:
                response = await self.client.messages.create(
                    model=self.model,
                    system=chunk_system_prompt,
                    max_tokens=4000,
                    messages=[
                        {
                            "role": "user",
                            "content": chunk_prompt
                        }
                    ]
                )
                
                chunk_analyses.append(response.content[0].text)
                
            except Exception as e:
                logger.error(f"Error analyzing chunk {i+1}: {e}")
                chunk_analyses.append(f"Error analyzing this section: {str(e)}")
        
        # Combine the analyses for a final assessment
        combined_prompt = f"""I've analyzed a research paper in multiple chunks due to its size. Here are my observations from each section:

{' '.join([f'SECTION {i+1}:\n{analysis}\n\n' for i, analysis in enumerate(chunk_analyses)])}

Based on these observations from different parts of the paper, provide a complete analysis according to the original criteria:
1. Innovation Score (0-10)
2. PoC Potential Score (0-10)
3. Wow Factor Score (0-10)
4. Implementation Complexity Score (0-10, lower is better)

And extract all requested information about the paper.
"""
        
        # Final analysis combining all chunks
        try:
            final_response = await self.client.messages.create(
                model=self.model,
                system=system_prompt,
                max_tokens=16000,
                messages=[
                    {
                        "role": "user", 
                        "content": combined_prompt
                    }
                ]
            )
            
            return final_response.content[0].text
            
        except Exception as e:
            logger.error(f"Error in final analysis: {e}")
            # Return whatever we've got
            return "ERROR in combined analysis. Partial results:\n\n" + "\n\n".join(chunk_analyses)
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF using PyPDF2.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text
        """
        try:
            import PyPDF2
            
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                text = ""
                for page_num in range(len(pdf_reader.pages)):
                    text += pdf_reader.pages[page_num].extract_text() + "\n\n"
                    
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text with PyPDF2: {e}")
            return ""
    
    def _extract_text_with_pdfplumber(self, pdf_path: Path) -> str:
        """
        Extract text from a PDF using pdfplumber (alternative method).
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text
        """
        try:
            import pdfplumber
            
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n\n"
                    
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text with pdfplumber: {e}")
            return ""
    
    def _split_text_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into chunks of roughly equal size.
        
        Args:
            text: Text to split
            chunk_size: Approximate size of each chunk in characters
            
        Returns:
            List of text chunks
        """
        chunks = []
        current_position = 0
        
        while current_position < len(text):
            # Find a good breaking point near the chunk_size
            chunk_end = min(current_position + chunk_size, len(text))
            
            # If we're not at the end, try to find a paragraph break
            if chunk_end < len(text):
                # Look for double newline (paragraph break)
                paragraph_break = text.rfind('\n\n', current_position, chunk_end)
                if paragraph_break != -1 and paragraph_break > current_position + chunk_size // 2:
                    chunk_end = paragraph_break + 2
                else:
                    # Look for single newline
                    newline = text.rfind('\n', current_position, chunk_end)
                    if newline != -1 and newline > current_position + chunk_size // 2:
                        chunk_end = newline + 1
                    else:
                        # Look for sentence end
                        sentence_end = max(
                            text.rfind('. ', current_position, chunk_end),
                            text.rfind('? ', current_position, chunk_end),
                            text.rfind('! ', current_position, chunk_end)
                        )
                        if sentence_end != -1 and sentence_end > current_position + chunk_size // 2:
                            chunk_end = sentence_end + 2
            
            # Add the chunk
            chunks.append(text[current_position:chunk_end])
            current_position = chunk_end
            
        return chunks
    
    def _process_response(self, content: str) -> Dict[str, Any]:
        """
        Process the Claude response to extract structured information.
        
        Args:
            content: Raw text response from Claude
            
        Returns:
            Dictionary containing structured analysis
        """
        # Create a basic structure for the analysis
        analysis = {
            "paper_title": "",
            "authors": [],
            "key_research_questions": [],
            "novel_methodology": "",
            "main_findings": "",
            "technical_details": "",
            "required_resources": "",
            "potential_applications": [],
            "implementation_challenges": [],
            "innovation_score": 0,
            "poc_potential_score": 0,
            "wow_factor_score": 0,
            "implementation_complexity_score": 0,
            "raw_analysis": content
        }
        
        try:
            # Extract scores using simple pattern matching
            for line in content.split("\n"):
                line = line.strip()
                
                # Match score patterns
                if "innovation score" in line.lower() or "innovation:" in line.lower():
                    score = self._extract_score(line)
                    if score is not None:
                        analysis["innovation_score"] = score
                
                elif "poc potential score" in line.lower() or "poc potential:" in line.lower() or "proof of concept potential:" in line.lower():
                    score = self._extract_score(line)
                    if score is not None:
                        analysis["poc_potential_score"] = score
                
                elif "wow factor score" in line.lower() or "wow factor:" in line.lower():
                    score = self._extract_score(line)
                    if score is not None:
                        analysis["wow_factor_score"] = score
                
                elif "implementation complexity score" in line.lower() or "implementation complexity:" in line.lower():
                    score = self._extract_score(line)
                    if score is not None:
                        analysis["implementation_complexity_score"] = score
                
                # Match paper title
                elif "paper title:" in line.lower() or "title:" in line.lower():
                    title_parts = line.split(":", 1)
                    if len(title_parts) > 1:
                        analysis["paper_title"] = title_parts[1].strip()
            
            # Calculate combined score with equal weights
            combined_score = (
                analysis["innovation_score"] * 0.25 +
                analysis["poc_potential_score"] * 0.25 +
                analysis["wow_factor_score"] * 0.25 +
                (10 - analysis["implementation_complexity_score"]) * 0.25  # Invert so lower complexity = higher score
            ) / 10  # Normalize to 0-1 range
            
            analysis["combined_score"] = combined_score
            
            logger.debug(f"Processed analysis with combined score: {combined_score}")
            return analysis
        
        except Exception as e:
            logger.error(f"Failed to process Claude response: {e}", exc_info=True)
            # Return the analysis with the raw response
            analysis["raw_analysis"] = content
            return analysis
    
    def _extract_score(self, text: str) -> Optional[float]:
        """
        Extract a numerical score from text.
        
        Args:
            text: Text containing a score
            
        Returns:
            Numerical score, or None if not found
        """
        import re
        
        # Look for patterns like "Score: 8/10" or "Score: 8"
        score_pattern = r"(?::|is|=)\s*(\d+(?:\.\d+)?)\s*(?:\/\s*\d+)?"
        match = re.search(score_pattern, text)
        
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                pass
        
        return None 