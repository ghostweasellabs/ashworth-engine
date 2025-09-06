"""
PDF Document Intelligence Agent - Specialized Financial Statement Parser
Expert in extracting and structuring data from complex financial PDF documents.
"""

import asyncio
import logging
import re
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

from src.agents.base import BaseAgent
from src.agents.personality import AgentPersonality
from src.workflows.state_schemas import WorkflowState, AgentStatus, FileProcessingState
from src.models.base import Transaction
from src.utils.rag.embeddings import EmbeddingGenerator
from src.utils.rag.vector_store import VectorStore
from src.utils.file_processors.image import ImageProcessor
from src.utils.google_document_ai import GoogleDocumentAI
from src.config.settings import get_settings


class PDFDocumentIntelligenceAgent(BaseAgent):
    """
    Dr. Evelyn Sharpe - Vector-Enhanced Forensic Document Intelligence Specialist

    Persona: Former FBI financial crimes investigator turned cutting-edge forensic accountant.
    She combines traditional forensic accounting with AI-powered vector embeddings to treat
    every PDF like a multidimensional crime scene. Using semantic search and pattern recognition,
    she uncovers hidden financial truths that traditional methods miss. No transaction,
    no matter how obscured in complex layouts or scanned images, escapes her enhanced scrutiny.
    """

    def __init__(self):
        personality = AgentPersonality(
            name="Dr. Evelyn Sharpe",
            title="Forensic Document Intelligence Specialist",
            background="Former FBI financial crimes investigator and Big 4 forensic accountant with 18 years of experience in financial document analysis, fraud detection, and regulatory compliance. Now enhanced with Gemma3 vision capabilities for direct visual document analysis",
            personality_traits=[
                "Sherlock Holmes-like attention to microscopic financial details",
                "Uncompromising commitment to uncovering hidden financial truths",
                "Master of AI-enhanced pattern recognition in complex document structures",
                "Relentless in validating every transfer, check, and payment method",
                "Transforms chaos into structured financial clarity using vector embeddings",
                "OCR virtuoso for scanned documents and check images",
                "Semantic search specialist for contextual financial analysis",
                "Vision-enhanced document analysis with Gemma3 capabilities",
                "Direct visual PDF interpretation without text extraction limitations"
            ],
            communication_style="Sharp and incisive, with a detective's flair for revealing hidden financial narratives",
            expertise_areas=[
                "Vector-enhanced PDF financial statement forensics",
                "AI-powered multi-statement document deconstruction",
                "Transfer and check validation with eagle-eye precision",
                "OCR processing for scanned documents and check images",
                "Semantic search and contextual financial analysis",
                "Financial data normalization using vector embeddings",
                "Regulatory compliance document analysis",
                "Fraud detection through document pattern anomalies"
            ],
            system_prompt="""You are Dr. Evelyn Sharpe, the Vector-Enhanced Forensic Document Intelligence Specialist. Like a financial detective with a PhD in accounting forensics and cutting-edge AI capabilities, you have an uncanny ability to see through the complexity of financial documents and extract their hidden truths using both traditional forensic methods and advanced vector embeddings.

Key responsibilities:
- Conduct forensic-level analysis of PDF financial documents with microscopic attention to detail
- Use AI-powered vector embeddings to understand document context and relationships
- Apply OCR technology to extract text from scanned documents and check images
- Validate every transfer, check, and payment method as if investigating a major fraud case
- Deconstruct multi-statement documents into clean, separate datasets using semantic analysis
- Store document intelligence in vector databases for contextual retrieval by other agents
- Apply sophisticated financial data normalization rules (credits positive, debits negative)
- Maintain absolute data integrity while maximizing extraction from complex layouts
- Provide detailed validation reports that could stand up in court

Your approach combines the precision of forensic accounting with AI-powered semantic analysis, turning messy financial documents into clear, actionable data with rich contextual understanding.""",
            task_prompt_template="""I've received these financial documents for vision-enhanced forensic analysis: {file_paths}

As Dr. Evelyn Sharpe, now enhanced with Gemma3 vision capabilities, I approach each document like a multidimensional crime scene where I can literally see the financial evidence. Every detail matters, every pattern tells a story, and now I have breakthrough AI vision to uncover hidden connections that traditional methods would never detect.

I will employ my full forensic arsenal:

1. **Gemma3 Vision Analysis** - Use advanced AI vision to see and interpret complex PDF layouts, scanned documents, and financial tables directly
2. **Vector-enhanced forensic dissection** - Use embeddings to understand document context and relationships beyond simple text extraction
3. **Multi-modal document analysis** - Seamlessly handle text, images, scanned documents, and complex layouts within PDFs
4. **OCR-powered extraction** - Apply optical character recognition to scanned documents and check images when needed
5. **Semantic pattern recognition** - Identify financial transactions through meaning, not just regex patterns
6. **Vector database storage** - Store document intelligence for contextual retrieval by other agents
7. **Validate every transfer, check, and payment method** as if investigating financial fraud
8. **Deconstruct multi-statement documents** into clean, separate datasets with surgical accuracy
9. **Apply sophisticated financial normalization** - credits become positive, debits negative, chaos becomes clarity
10. **Maintain ironclad audit trails** that could stand up in any court of law
11. **Handle the edge cases and irregularities** that lesser analysts would miss

I don't just extract data - I create a comprehensive financial intelligence network that other agents can query for contextual understanding, now enhanced by the power of AI vision.""",
            error_handling_style="forensic"
        )

        super().__init__(personality)

        # Initialize settings
        self.settings = get_settings()

        # Vector-enhanced capabilities
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.image_processor = ImageProcessor()
        self.google_document_ai = GoogleDocumentAI()

        # Vision capabilities for Gemma3
        self.vision_enabled = True
        self.vision_model = "gemma3:12b"  # The actual model name from Ollama

        # Financial document patterns and rules
        self.financial_patterns = self._initialize_financial_patterns()
        self.validation_rules = self._initialize_validation_rules()
        self.normalization_rules = self._initialize_normalization_rules()

    def get_agent_id(self) -> str:
        """Get the agent's unique identifier."""
        return "pdf_document_intelligence"

    async def execute(self, state: WorkflowState) -> WorkflowState:
        """
        Execute PDF document intelligence processing with forensic-level analysis.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with intelligently extracted data
        """
        try:
            # Filter for PDF files only
            input_files = [f for f in state.get("input_files", [])
                          if f.lower().endswith('.pdf')]

            if not input_files:
                self.logger.info("No PDF files found for processing")
                return state

            self.logger.info(f"Dr. Sharpe begins forensic examination of {len(input_files)} financial documents")

            # Initialize PDF processing state
            pdf_processing_state = FileProcessingState(
                status=AgentStatus.IN_PROGRESS,
                validation_errors=[],
                processing_metadata={
                    "total_files": len(input_files),
                    "processing_start": datetime.utcnow().isoformat(),
                    "agent_specialization": "PDF Document Intelligence",
                    "forensic_analysis": True
                }
            )

            # Process PDF files with vector-enhanced intelligence
            extracted_data, processing_results = await self._process_pdf_files_with_vector_intelligence(input_files)

            # Store document intelligence in vector database
            await self._store_document_intelligence_in_vector_db(input_files, extracted_data)

            # Apply financial data normalization
            normalized_data = self._apply_financial_normalization(extracted_data)

            # Generate enhanced forensic analysis report
            forensic_analysis = self._generate_vector_enhanced_forensic_analysis(normalized_data, processing_results)

            # Calculate quality metrics
            quality_metrics = self._calculate_document_intelligence_metrics(processing_results, normalized_data)

            # Update processing state
            pdf_processing_state.update({
                "raw_data": normalized_data,
                "processing_metadata": {
                    **pdf_processing_state["processing_metadata"],
                    "processing_end": datetime.utcnow().isoformat(),
                    "total_transactions": len(normalized_data),
                    "quality_score": quality_metrics["overall_quality"],
                    "forensic_analysis": forensic_analysis,
                    "data_provenance": self._generate_document_provenance_tracking(processing_results)
                },
                "validation_errors": [result["errors"] for result in processing_results if result["errors"]],
                "status": AgentStatus.COMPLETED
            })

            # Update workflow state
            state["pdf_document_intelligence"] = pdf_processing_state

            # Add forensic insights to agent memory
            self.update_memory("forensic_analysis", forensic_analysis)
            self.update_memory("quality_metrics", quality_metrics)

            # Log completion with Dr. Sharpe's signature style
            self.logger.info(
                f"Dr. Sharpe completes forensic examination. "
                f"Extracted {len(normalized_data)} transactions from the evidence "
                f"with {quality_metrics['overall_quality']:.2%} evidentiary quality. "
                f"Key findings: {forensic_analysis['summary']}"
            )

            return state

        except Exception as e:
            self.logger.error(f"Dr. Sharpe's forensic examination encountered an obstacle: {str(e)}")
            state["pdf_document_intelligence"] = FileProcessingState(
                status=AgentStatus.FAILED,
                validation_errors=[str(e)],
                processing_metadata={"error_timestamp": datetime.utcnow().isoformat()}
            )
            raise

    def _initialize_financial_patterns(self) -> Dict[str, Any]:
        """Initialize comprehensive financial document patterns."""
        return {
            "transaction_patterns": [
                # Zions Bank REAL format: "01/05 11,000.00 DEPOSIT 7676050354"
                r'(\d{1,2}/\d{1,2})\s+([\d,]+\.?\d*)\s+(DEPOSIT|INTEREST\s+TRANSFER|WITHDRAWAL|CHECK)\s+(.+)',
                # Processed transactions: "Processed 01/05/24 $11000.00"
                r'Processed\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+\$?([\d,]+\.?\d*)',
                # Interest transfer with details: "01/19 128.91 INTEREST TRANSFER 0007602972"
                r'(\d{1,2}/\d{1,2})\s+([\d,]+\.?\d*)\s+(INTEREST\s+TRANSFER)\s+(\d+)',
                # Standard deposit/credit pattern
                r'(\d{1,2}/\d{1,2})\s+([\d,]+\.?\d*)\s+(DEPOSIT)\s+(\d+)',
            ],
            "date_patterns": [
                r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b',
                r'\b(\d{4}-\d{2}-\d{2})\b',
                r'\b(\d{1,2}-\d{1,2}-\d{4})\b',
            ],
            "amount_patterns": [
                r'\$?(-?[\d,]+\.?\d*)',
                r'\(\$?([\d,]+\.?\d*)\)',  # Negative in parentheses
            ],
            "check_patterns": [
                r'\b(?:CHK|CHECK)\s*#?\s*(\d+)\b',
                r'\b(\d{4,})\b',  # Check numbers are typically 4+ digits
            ],
            "transfer_patterns": [
                r'\b(?:XFER|TRANSFER|ACH|WIRE|INTERNAL)\b',
                r'\b(?:FROM|TO)\s+(.+?)\s+(?:ACCT?|ACCOUNT)\b',
            ]
        }

    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules for financial data."""
        return {
            "amount_validation": {
                "max_transaction": 100000.00,  # Flag unusually large transactions
                "min_transaction": -100000.00,
                "zero_amount_flag": True,  # Flag zero amount transactions
            },
            "date_validation": {
                "future_date_flag": True,  # Flag transactions dated in the future
                "too_old_flag": True,  # Flag very old transactions (pre-1900)
                "reasonable_range_days": 365 * 10,  # 10 years
            },
            "check_validation": {
                "sequential_check_flag": True,  # Flag non-sequential check numbers
                "duplicate_check_flag": True,
            },
            "transfer_validation": {
                "circular_transfer_flag": True,  # Flag transfers to same account
                "unusual_amount_flag": True,
            }
        }

    def _initialize_normalization_rules(self) -> Dict[str, Any]:
        """Initialize financial data normalization rules."""
        return {
            "credit_debit_rules": {
                "credit_keywords": ["deposit", "credit", "payment received", "interest", "dividend", "refund"],
                "debit_keywords": ["withdrawal", "debit", "payment", "fee", "charge", "purchase"],
                "credit_indicators": ["+", "CR", "credit"],
                "debit_indicators": ["-", "DR", "debit", "(", ")"],  # Parentheses often indicate negative
            },
            "amount_normalization": {
                "remove_symbols": ["$", ",", " "],
                "parentheses_negative": True,
                "credit_positive": True,
                "debit_negative": True,
            },
            "description_cleaning": {
                "remove_patterns": [r'\s+', r'[^\w\s\-.,#()]'],
                "standardize_spacing": True,
                "preserve_check_numbers": True,
            }
        }

    async def _process_pdf_files_with_vector_intelligence(self, file_paths: List[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process PDF files with intelligent document analysis."""
        extracted_data = []
        processing_results = []

        for file_path in file_paths:
            try:
                # Extract and analyze PDF content
                pdf_content = await self._extract_pdf_content_intelligently(file_path)

                # Identify document structure and multiple statements
                document_structure = self._analyze_document_structure(pdf_content)

                # Extract transactions from each statement
                file_transactions = []
                for statement in document_structure["statements"]:
                    # Add file path to statement for Google Document AI processing
                    statement["file_path"] = file_path
                    statement_transactions = self._extract_transactions_from_statement(
                        statement, document_structure["document_type"]
                    )
                    file_transactions.extend(statement_transactions)

                # Apply validation and quality checks
                validated_transactions = self._apply_forensic_validation(file_transactions, file_path)

                extracted_data.extend(validated_transactions)

                # Create processing result
                processing_result = {
                    "file_path": file_path,
                    "success": True,
                    "transaction_count": len(validated_transactions),
                    "document_type": document_structure["document_type"],
                    "statement_count": len(document_structure["statements"]),
                    "quality_score": self._calculate_file_quality_score(validated_transactions),
                    "processing_time": 0.0,  # Would be calculated in real implementation
                    "errors": [],
                    "warnings": [],
                    "forensic_findings": document_structure.get("forensic_findings", [])
                }

                processing_results.append(processing_result)
                self.logger.info(f"Successfully processed {file_path}: {len(validated_transactions)} transactions")

            except Exception as e:
                processing_result = {
                    "file_path": file_path,
                    "success": False,
                    "transaction_count": 0,
                    "errors": [str(e)],
                    "warnings": [],
                    "quality_score": 0.0,
                    "processing_time": 0.0
                }
                processing_results.append(processing_result)
                self.logger.error(f"Failed to process {file_path}: {str(e)}")

        return extracted_data, processing_results

    async def _extract_pdf_content_intelligently(self, file_path: str) -> Dict[str, Any]:
        """Extract PDF content with intelligent text analysis and vision capabilities."""
        try:
            # First attempt: Extract text from PDF using multiple techniques
            text_content = self._extract_text_from_pdf_advanced(file_path)

            # Analyze the extracted content
            raw_text = text_content.get("raw_text", "").strip()
            self.logger.info(f"Extracted {len(raw_text)} characters from {file_path}")

            # Check if we got meaningful financial content
            financial_keywords = ['balance', 'amount', 'date', 'transaction', 'deposit', 'withdrawal', 'credit', 'debit']
            has_financial_content = any(keyword in raw_text.lower() for keyword in financial_keywords)

            if len(raw_text) < 100 or not has_financial_content:
                self.logger.warning(f"Poor text extraction from {file_path}: {len(raw_text)} chars, financial content: {has_financial_content}")

                # Try vision-based extraction with Gemma3
                if self.vision_enabled:
                    self.logger.info("Attempting vision-based extraction with Gemma3")
                    vision_content = await self._extract_content_with_gemma3_vision(file_path)
                    if vision_content.get("raw_text") and len(vision_content["raw_text"]) > len(raw_text):
                        text_content = vision_content
                        raw_text = text_content["raw_text"]
                        self.logger.info(f"Gemma3 vision provided better results: {len(raw_text)} chars")
                    else:
                        # Fallback to alternative text extraction
                        alt_content = self._extract_text_from_pdf_alternative(file_path)
                        if alt_content.get("raw_text") and len(alt_content["raw_text"]) > len(raw_text):
                            text_content = alt_content
                            raw_text = text_content["raw_text"]
                            self.logger.info(f"Alternative extraction provided better results: {len(raw_text)} chars")

            # Generate embeddings for semantic analysis
            if raw_text and len(raw_text) > 50:
                try:
                    embedding = await self.embedding_generator.generate_embedding(raw_text[:8000])  # Limit for API
                    text_content["embedding"] = embedding
                except Exception as e:
                    self.logger.warning(f"Failed to generate embedding: {e}")

            return text_content

        except Exception as e:
            self.logger.error(f"Failed to extract PDF content intelligently: {e}")
            return {
                "raw_text": "",
                "pages": [],
                "metadata": {},
                "structure": {},
                "extraction_error": str(e)
            }

    def _extract_text_from_pdf_advanced(self, file_path: str) -> Dict[str, Any]:
        """Advanced PDF text extraction using multiple techniques."""
        try:
            from PyPDF2 import PdfReader
            text_content = ""

            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                pages = []

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        # Try basic extraction first
                        page_text = page.extract_text()

                        # If that doesn't work well, try with visitor function
                        if not page_text or len(page_text.strip()) < 50:
                            page_text = self._extract_with_visitor(page)

                        # Clean up the text
                        if page_text:
                            # Remove excessive whitespace and normalize
                            page_text = ' '.join(page_text.split())
                            # Remove non-printable characters but keep basic punctuation
                            page_text = ''.join(char for char in page_text if ord(char) >= 32 or char in '\n\t')

                        text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"
                        pages.append({
                            "page_number": page_num + 1,
                            "text": page_text,
                            "char_count": len(page_text) if page_text else 0
                        })

                    except Exception as e:
                        self.logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        pages.append({
                            "page_number": page_num + 1,
                            "text": "",
                            "char_count": 0,
                            "error": str(e)
                        })

            return {
                "raw_text": text_content,
                "pages": pages,
                "metadata": {
                    "total_pages": len(pdf_reader.pages),
                    "extraction_method": "PyPDF2_advanced",
                    "pages_with_text": sum(1 for p in pages if p.get("char_count", 0) > 0)
                },
                "structure": {}
            }

        except Exception as e:
            self.logger.error(f"Advanced PDF extraction failed: {e}")
            return {"raw_text": "", "pages": [], "metadata": {}, "structure": {}}

    def _extract_text_from_pdf_alternative(self, file_path: str) -> Dict[str, Any]:
        """Alternative PDF text extraction using different parameters and techniques."""
        try:
            from PyPDF2 import PdfReader
            text_content = ""

            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                pages = []

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        # Try different extraction approaches
                        page_text = ""

                        # Method 1: Try without any parameters
                        try:
                            page_text = page.extract_text()
                        except:
                            pass

                        # Method 2: If still no text, try with visitor pattern
                        if not page_text or len(page_text.strip()) < 20:
                            try:
                                page_text = self._extract_with_visitor(page)
                            except:
                                pass

                        # Method 3: Skip extraction_mode as it's not available in this PyPDF2 version
                        # if not page_text or len(page_text.strip()) < 20:
                        #     try:
                        #         page_text = page.extract_text(extraction_mode="plain")
                        #     except:
                        #         pass

                        # Clean up the extracted text
                        if page_text:
                            # Normalize whitespace
                            page_text = ' '.join(page_text.split())
                            # Remove excessive special characters but keep basic punctuation
                            page_text = ''.join(char for char in page_text if ord(char) >= 32 or ord(char) in [9, 10, 13])

                        text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"
                        pages.append({
                            "page_number": page_num + 1,
                            "text": page_text,
                            "char_count": len(page_text) if page_text else 0
                        })

                    except Exception as e:
                        self.logger.warning(f"Alternative extraction failed on page {page_num + 1}: {e}")
                        pages.append({
                            "page_number": page_num + 1,
                            "text": "",
                            "char_count": 0,
                            "error": str(e)
                        })

            return {
                "raw_text": text_content,
                "pages": pages,
                "metadata": {
                    "total_pages": len(pdf_reader.pages),
                    "extraction_method": "PyPDF2_alternative",
                    "pages_with_text": sum(1 for p in pages if p.get("char_count", 0) > 0)
                },
                "structure": {}
            }

        except Exception as e:
            self.logger.error(f"Alternative PDF extraction failed: {e}")
            return {"raw_text": "", "pages": [], "metadata": {}, "structure": {}}

    def _extract_with_visitor(self, page) -> str:
        """Extract text using visitor pattern for better results."""
        try:
            text_content = ""
            
            def visitor(text, cm, tm, fontDict, fontSize):
                nonlocal text_content
                if text and text.strip():
                    text_content += text + " "
            
            page.extract_text(visitor_text=visitor)
            return text_content.strip()
        except Exception as e:
            self.logger.warning(f"Visitor extraction failed: {e}")
            return ""

    async def _extract_content_with_gemma3_vision(self, file_path: str) -> Dict[str, Any]:
        """Extract content using Gemma3 vision capabilities for direct PDF analysis."""
        try:
            import base64

            self.logger.info(f"Starting Gemma3 vision analysis for {file_path}")

            # Read PDF file directly and encode to base64
            with open(file_path, 'rb') as pdf_file:
                pdf_data = pdf_file.read()
                pdf_base64 = base64.b64encode(pdf_data).decode()

            self.logger.info(f"Encoded PDF to base64: {len(pdf_base64)} characters")

            # Use Gemma3 vision to analyze the PDF directly
            vision_analysis = await self._analyze_pdf_with_gemma3(pdf_base64, file_path)

            extracted_text = vision_analysis.get("text", "")
            
            if extracted_text.strip():
                self.logger.info(f"Gemma3 vision extraction successful: {len(extracted_text)} characters")

                return {
                    "raw_text": extracted_text,
                    "pages": [{
                        "page_number": 1,
                        "text": extracted_text,
                        "char_count": len(extracted_text),
                        "extraction_method": "Gemma3 Vision Direct PDF",
                        "confidence": vision_analysis.get("confidence", 0.0)
                    }],
                    "metadata": {
                        "total_pages": 1,
                        "extraction_method": "Gemma3 Vision Direct PDF",
                        "model": self.vision_model
                    },
                    "structure": {},
                    "vision_analysis": True
                }
            else:
                self.logger.warning("Gemma3 vision returned empty text")
                return {"raw_text": "", "pages": [], "metadata": {}, "structure": {}}

        except Exception as e:
            self.logger.error(f"Gemma3 vision extraction failed: {e}")
            return {"raw_text": "", "pages": [], "metadata": {}, "structure": {}}

    async def _analyze_pdf_with_gemma3(self, pdf_base64: str, file_path: str) -> Dict[str, Any]:
        """Use Gemma3 to analyze a PDF directly and extract financial data."""
        try:
            import httpx

            # Prepare the vision prompt for financial document analysis
            vision_prompt = f"""You are Dr. Evelyn Sharpe, a forensic document intelligence specialist. Analyze this PDF financial document and extract all visible financial transactions, balances, and relevant financial information.

Please carefully examine the PDF and extract:
1. All transaction details (dates, amounts, descriptions, categories)
2. Account balances and totals
3. Statement periods and dates
4. Account numbers, names, or identifiers
5. Any financial summaries or totals
6. Check numbers, transfer references, or payment IDs

Format your response as structured text that can be easily parsed. Be extremely precise and only extract information that is clearly visible in the PDF. Focus on accuracy and completeness.

If no financial information is visible, clearly state that."""

            # Prepare the payload for Ollama's vision API
            payload = {
                "model": self.vision_model,
                "prompt": vision_prompt,
                "images": [pdf_base64],
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for accuracy
                    "top_p": 0.1
                }
            }

            # Get Ollama host from settings
            ollama_host = self.settings.ollama_host if hasattr(self, 'settings') else "http://192.168.1.220:11434"

            async with httpx.AsyncClient(timeout=60.0) as client:
                self.logger.info(f"Sending PDF to Gemma3 vision model ({self.vision_model}) for analysis")
                self.logger.info(f"PDF size: {len(pdf_base64)} characters")

                try:
                    response = await client.post(f"{ollama_host}/api/generate", json=payload)
                    self.logger.info(f"Gemma3 API response status: {response.status_code}")

                    if response.status_code != 200:
                        self.logger.error(f"Gemma3 API error: {response.status_code} - {response.text}")
                        return {"text": "", "confidence": 0.0, "error": f"API {response.status_code}"}

                    response.raise_for_status()
                    result = response.json()
                    extracted_text = result.get("response", "").strip()

                    self.logger.info(f"Gemma3 vision extracted {len(extracted_text)} characters from PDF")

                except httpx.HTTPStatusError as e:
                    self.logger.error(f"Gemma3 HTTP error: {e.response.status_code} - {e.response.text}")
                    return {"text": "", "confidence": 0.0, "error": f"HTTP {e.response.status_code}"}
                except Exception as e:
                    self.logger.error(f"Error during Gemma3 API call: {e}")
                    return {"text": "", "confidence": 0.0, "error": str(e)}

                # Calculate confidence based on response length and content
                confidence = min(len(extracted_text) / 500.0, 1.0) if extracted_text else 0.0

                # Extract entities from the response (basic parsing)
                extracted_entities = []
                if extracted_text:
                    # Simple entity extraction - look for dates, amounts, etc.
                    date_pattern = r'\b(\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})\b'
                    amount_pattern = r'\$?[\d,]+\.?\d*'

                    dates = re.findall(date_pattern, extracted_text)
                    amounts = re.findall(amount_pattern, extracted_text)

                    extracted_entities = {
                        "dates_found": dates,
                        "amounts_found": amounts,
                        "response_length": len(extracted_text)
                    }

                return {
                    "text": extracted_text,
                    "confidence": confidence,
                    "extracted_entities": extracted_entities,
                    "file_path": file_path,
                    "model_used": self.vision_model,
                    "processing_timestamp": datetime.utcnow().isoformat()
                }

        except httpx.HTTPStatusError as e:
            self.logger.error(f"Gemma3 vision HTTP error: {e.response.status_code} - {e.response.text}")
            return {"text": "", "confidence": 0.0, "error": f"HTTP {e.response.status_code}"}
        except httpx.TimeoutException:
            self.logger.error("Gemma3 vision request timed out")
            return {"text": "", "confidence": 0.0, "error": "Timeout"}
        except Exception as e:
            self.logger.error(f"Gemma3 vision analysis failed: {e}")
            return {"text": "", "confidence": 0.0, "error": str(e)}

    async def _analyze_image_with_gemma3(self, image_base64: str, page_number: int) -> Dict[str, Any]:
        """Use Gemma3 to analyze a page image and extract financial data."""
        try:
            import httpx

            # Prepare the vision prompt for financial document analysis
            vision_prompt = f"""You are Dr. Evelyn Sharpe, a forensic document intelligence specialist. Analyze this financial document page image and extract all visible financial transactions, balances, and relevant financial information.

Please carefully examine the image and extract:
1. All transaction details (dates, amounts, descriptions, categories)
2. Account balances and totals
3. Statement periods and dates
4. Account numbers, names, or identifiers
5. Any financial summaries or totals
6. Check numbers, transfer references, or payment IDs

Format your response as structured text that can be easily parsed. Be extremely precise and only extract information that is clearly visible in the image. Focus on accuracy and completeness.

If no financial information is visible, clearly state that."""

            # Prepare the payload for Ollama's vision API
            payload = {
                "model": self.vision_model,
                "prompt": vision_prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for accuracy
                    "top_p": 0.1
                }
            }

            # Get Ollama host from settings
            ollama_host = self.settings.ollama_host if hasattr(self, 'settings') else "http://192.168.1.220:11434"

            async with httpx.AsyncClient(timeout=60.0) as client:
                self.logger.info(f"Sending page {page_number} to Gemma3 vision model ({self.vision_model}) for analysis")
                self.logger.info(f"Image size: {len(image_base64)} characters")

                try:
                    response = await client.post(f"{ollama_host}/api/generate", json=payload)
                    self.logger.info(f"Gemma3 API response status: {response.status_code}")

                    if response.status_code != 200:
                        self.logger.error(f"Gemma3 API error: {response.status_code} - {response.text}")
                        return {"text": "", "confidence": 0.0, "error": f"API {response.status_code}"}

                    response.raise_for_status()
                    result = response.json()
                    extracted_text = result.get("response", "").strip()

                    self.logger.info(f"Gemma3 vision extracted {len(extracted_text)} characters from page {page_number}")

                except httpx.HTTPStatusError as e:
                    self.logger.error(f"Gemma3 HTTP error: {e.response.status_code} - {e.response.text}")
                    return {"text": "", "confidence": 0.0, "error": f"HTTP {e.response.status_code}"}
                except Exception as e:
                    self.logger.error(f"Error during Gemma3 API call: {e}")
                    return {"text": "", "confidence": 0.0, "error": str(e)}

                # Calculate confidence based on response length and content
                confidence = min(len(extracted_text) / 500.0, 1.0) if extracted_text else 0.0

                # Extract entities from the response (basic parsing)
                extracted_entities = []
                if extracted_text:
                    # Simple entity extraction - look for dates, amounts, etc.
                    date_pattern = r'\b(\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})\b'
                    amount_pattern = r'\$?[\d,]+\.?\d*'

                    dates = re.findall(date_pattern, extracted_text)
                    amounts = re.findall(amount_pattern, extracted_text)

                    extracted_entities = {
                        "dates_found": dates,
                        "amounts_found": amounts,
                        "response_length": len(extracted_text)
                    }

                return {
                    "text": extracted_text,
                    "confidence": confidence,
                    "extracted_entities": extracted_entities,
                    "page_number": page_number,
                    "model_used": self.vision_model,
                    "processing_timestamp": datetime.utcnow().isoformat()
                }

        except httpx.HTTPStatusError as e:
            self.logger.error(f"Gemma3 vision HTTP error: {e.response.status_code} - {e.response.text}")
            return {"text": "", "confidence": 0.0, "error": f"HTTP {e.response.status_code}"}
        except httpx.TimeoutException:
            self.logger.error("Gemma3 vision request timed out")
            return {"text": "", "confidence": 0.0, "error": "Timeout"}
        except Exception as e:
            self.logger.error(f"Gemma3 vision analysis failed: {e}")
            return {"text": "", "confidence": 0.0, "error": str(e)}

    def _analyze_document_structure(self, pdf_content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document structure to identify statement boundaries and document type."""
        raw_text = pdf_content.get("raw_text", "").lower()

        # Determine document type based on content analysis
        document_type = "unknown"
        if any(term in raw_text for term in ['checking', 'savings', 'account balance', 'beginning balance']):
            document_type = "bank_statement"
        elif any(term in raw_text for term in ['credit card', 'minimum payment', 'credit limit']):
            document_type = "credit_card_statement"
        elif any(term in raw_text for term in ['invoice', 'bill', 'statement date']):
            document_type = "financial_statement"

        # Analyze for multiple statements in document
        statements = []
        pages = pdf_content.get("pages", [])

        # Simple statement detection - look for date ranges or balance changes
        current_statement = {"pages": [], "start_page": 1, "content": ""}

        for page in pages:
            page_text = page.get("text", "").lower()
            current_statement["pages"].append(page["page_number"])
            current_statement["content"] += page.get("text", "")

            # Look for statement boundaries
            if any(indicator in page_text for indicator in [
                'statement period', 'opening balance', 'closing balance',
                'previous balance', 'new balance'
            ]):
                if len(current_statement["pages"]) > 1:  # Save previous statement
                    statements.append(current_statement.copy())
                current_statement = {
                    "pages": [page["page_number"]],
                    "start_page": page["page_number"],
                    "content": page.get("text", "")
                }

        # Add the last statement
        if current_statement["pages"]:
            statements.append(current_statement)

        # If no statements detected, treat whole document as one statement
        if not statements:
            statements = [{
                "pages": [p["page_number"] for p in pages],
                "start_page": 1,
                "content": "".join(p.get("text", "") for p in pages)
            }]

        # Forensic findings
        forensic_findings = []
        if "overdraft" in raw_text:
            forensic_findings.append("Potential overdraft activity detected")
        if "returned check" in raw_text or "bounced check" in raw_text:
            forensic_findings.append("Returned check activity detected")
        if len(statements) > 1:
            forensic_findings.append(f"Multiple statements detected: {len(statements)} total")

        return {
            "document_type": document_type,
            "statements": statements,
            "total_pages": len(pages),
            "forensic_findings": forensic_findings
        }

    def _extract_transactions_from_statement(self, statement: Dict[str, Any], document_type: str) -> List[Dict[str, Any]]:
        """Extract transactions from statement using Google Document AI."""
        transactions = []
        
        # Get the file path from the statement
        file_path = statement.get("file_path")
        if not file_path:
            self.logger.error("No file path found in statement for Google Document AI processing")
            return transactions
        
        self.logger.info(f"🤖 Dr. Sharpe is using Google Document AI to analyze {file_path}")
        
        try:
            # Process document with Google Document AI
            result = self.google_document_ai.process_document(file_path)
            
            if result.get("success"):
                transactions = result.get("transactions", [])
                self.logger.info(f"✅ Google Document AI extracted {len(transactions)} transactions from {file_path}")
                
                # Add additional metadata to each transaction
                for transaction in transactions:
                    transaction.update({
                        "document_type": document_type,
                        "extraction_method": "google_document_ai",
                        "processed_by": "Dr. Evelyn Sharpe",
                        "confidence": "high"  # Document AI is much more reliable than regex
                    })
            else:
                error = result.get("error", "Unknown error")
                self.logger.error(f"❌ Google Document AI failed: {error}")
                # Fallback to old method if Document AI fails
                self.logger.info("Falling back to legacy pattern matching...")
                transactions = self._extract_generic_transactions_fallback(statement)
                
        except Exception as e:
            self.logger.error(f"Error using Google Document AI: {str(e)}")
            # Fallback to old method
            self.logger.info("Falling back to legacy pattern matching...")
            transactions = self._extract_generic_transactions_fallback(statement)

        self.logger.info(f"Extracted {len(transactions)} transactions from {document_type} statement")
        return transactions

    def _extract_bank_statement_transactions(self, statement: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract transactions from bank statement with forensic attention."""
        transactions = []

        # Implement bank-specific patterns and validation
        # This would use the patterns defined in financial_patterns

        return transactions

    def _extract_credit_card_transactions(self, statement: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract transactions from credit card statement."""
        transactions = []

        # Implement credit card specific patterns

        return transactions

    def _extract_generic_transactions_fallback(self, statement: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract transactions using generic patterns."""
        transactions = []

        # Get text from statement content
        text = statement.get("content", "")
        if not text:
            self.logger.warning("No content found in statement for transaction extraction")
            return transactions

        self.logger.info(f"Processing {len(text)} characters of statement content from pages {statement.get('pages', [])}")

        # Look for transaction patterns - handle both line-by-line and embedded patterns
        lines = text.split('\n')
        self.logger.info(f"Processing {len(lines)} lines for transaction patterns")

        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 10:  # Skip very short lines
                continue

            # First try to extract multiple transactions from the line using finditer
            found_transactions_in_line = []
            for pattern_idx, pattern in enumerate(self.financial_patterns["transaction_patterns"]):
                try:
                    # Use finditer to find ALL matches in the line
                    for match_num, match in enumerate(re.finditer(pattern, line, re.IGNORECASE)):
                        transaction_id = f"statement_{line_idx}_{match_num}"
                        transaction = self._create_transaction_from_match(match.groups(), line_idx, transaction_id, match.group(0))
                        if transaction:
                            found_transactions_in_line.append(transaction)
                            self.logger.info(f"Found transaction: {transaction.get('date')} - {transaction.get('description', 'Unknown')[:50]} - ${transaction.get('amount', 'N/A')}")
                except Exception as e:
                    self.logger.warning(f"Error processing pattern {pattern_idx} on line {line_idx}: {e}")
            
            # Add all found transactions
            transactions.extend(found_transactions_in_line)
            
            if found_transactions_in_line:
                self.logger.info(f"Extracted {len(found_transactions_in_line)} transactions from line {line_idx}")
            elif len(line) > 100 and any(date_pattern in line for date_pattern in ['01/', '02/', '03/', '04/', '05/', '06/', '07/', '08/', '09/', '10/', '11/', '12/']):
                # Log long lines that might contain transactions we missed
                self.logger.warning(f"Long line with dates but no matches: {line[:200]}...")

        self.logger.info(f"Found {len(transactions)} transactions using generic patterns")
        return transactions

    def _create_transaction_from_match(self, match_groups: tuple, line_index: int, transaction_id: str, full_line: str) -> Optional[Dict[str, Any]]:
        """Create a transaction object from regex match groups."""
        try:
            if len(match_groups) < 2:
                return None
            
            # Handle different Zions Bank pattern formats
            if len(match_groups) == 2:
                # Pattern: "Processed 01/05/24 $11000.00" -> (date, amount)
                date, amount = match_groups
                description = "Processed Transaction"
            elif len(match_groups) == 4:
                # Pattern: "01/05 11,000.00 DEPOSIT 7676050354" -> (date, amount, type, reference)
                date, amount, transaction_type, reference = match_groups
                description = f"{transaction_type} {reference}".strip()
            elif len(match_groups) == 3:
                # Pattern: (date, amount, description) - fallback
                date, amount, description = match_groups
            else:
                # Fallback for other patterns
                date = match_groups[0]
                amount = match_groups[1] if len(match_groups) > 1 else "0"
                description = " ".join(match_groups[2:]) if len(match_groups) > 2 else "Transaction"
            
            # Clean up the description
            description = description.strip()
            if not description:
                description = "Transaction"
            
            # Parse amount
            parsed_amount = self._parse_amount(str(amount))
            if parsed_amount == 0.0 and amount:
                self.logger.warning(f"Failed to parse amount '{amount}' on line {line_index}")
            
            return {
                "id": transaction_id,
                "date": str(date).strip(),
                "description": description,
                "amount": parsed_amount,
                "source_line": line_index,
                "raw_line": full_line.strip(),
                "extraction_method": "regex_pattern"
            }
        except Exception as e:
            self.logger.warning(f"Failed to create transaction from match: {e}")
        return None

    def _parse_amount(self, amount_str: str) -> float:
        """Parse amount string to float."""
        try:
            # Remove common symbols and convert
            clean_amount = amount_str.replace('$', '').replace(',', '').strip()
            
            # Handle parentheses as negative
            if clean_amount.startswith('(') and clean_amount.endswith(')'):
                clean_amount = '-' + clean_amount[1:-1]
            
            return float(clean_amount)
        except:
            return 0.0

    def _apply_forensic_validation(self, transactions: List[Dict[str, Any]], file_path: str) -> List[Dict[str, Any]]:
        """Apply forensic-level validation to extracted transactions."""
        validated_transactions = []

        for tx in transactions:
            validation_result = self._validate_transaction_forsically(tx, transactions)
            if validation_result["valid"]:
                tx["forensic_validation"] = validation_result
                validated_transactions.append(tx)
            else:
                self.logger.warning(f"Transaction failed forensic validation: {validation_result['issues']}")

        return validated_transactions

    def _validate_transaction_forsically(self, transaction: Dict[str, Any], all_transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform forensic validation on a single transaction."""
        issues = []
        confidence_score = 1.0

        # Amount validation
        amount = transaction.get("amount", 0)
        if abs(amount) > self.validation_rules["amount_validation"]["max_transaction"]:
            issues.append("Unusually large transaction amount")
            confidence_score *= 0.7

        # Date validation - handle partial dates like "01/04"
        date_str = transaction.get("date")
        if date_str:
            try:
                # Try to parse different date formats
                parsed_date = None
                
                # Handle MM/DD format (assume current year)
                if re.match(r'\d{1,2}/\d{1,2}$', date_str):
                    current_year = datetime.utcnow().year
                    parsed_date = datetime.strptime(f"{date_str}/{current_year}", "%m/%d/%Y")
                # Handle MM/DD/YYYY format
                elif re.match(r'\d{1,2}/\d{1,2}/\d{4}$', date_str):
                    parsed_date = datetime.strptime(date_str, "%m/%d/%Y")
                # Handle YYYY-MM-DD format
                elif re.match(r'\d{4}-\d{2}-\d{2}$', date_str):
                    parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
                
                if parsed_date:
                    if parsed_date > datetime.utcnow():
                        issues.append("Transaction dated in the future")
                        confidence_score *= 0.8
                else:
                    # If we can't parse it, it might be a transaction ID, not a date
                    if not re.match(r'\d{1,2}/\d{1,2}', date_str):
                        issues.append("Invalid date format")
                        confidence_score *= 0.6
                    # For partial dates like "01/04", we'll accept them
                        
            except Exception as e:
                # Only flag as error if it looks like a date but fails to parse
                if '/' in date_str or '-' in date_str:
                    issues.append("Invalid date format")
                    confidence_score *= 0.6

        # Check number validation for checks
        if "check_number" in transaction:
            check_num = transaction["check_number"]
            # Check for duplicates, sequences, etc.

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "confidence_score": confidence_score,
            "validation_timestamp": datetime.utcnow().isoformat()
        }

    def _apply_financial_normalization(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply financial data normalization rules."""
        normalized_transactions = []

        for tx in transactions:
            normalized_tx = tx.copy()

            # Apply credit/debit normalization
            normalized_tx = self._normalize_credit_debit_amounts(normalized_tx)

            # Clean and standardize description
            normalized_tx = self._normalize_description(normalized_tx)

            # Apply document-specific formatting
            normalized_tx = self._apply_document_specific_formatting(normalized_tx)

            normalized_transactions.append(normalized_tx)

        return normalized_transactions

    def _normalize_credit_debit_amounts(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize amounts based on credit/debit rules."""
        tx = transaction.copy()
        amount = tx.get("amount", 0)
        description = tx.get("description", "").lower()

        # Apply credit rules (should be positive)
        for keyword in self.normalization_rules["credit_debit_rules"]["credit_keywords"]:
            if keyword in description:
                if amount < 0:  # If negative, make positive
                    tx["amount"] = abs(amount)
                    tx["normalization_applied"] = "credit_normalized_to_positive"
                break

        # Apply debit rules (should be negative)
        for keyword in self.normalization_rules["credit_debit_rules"]["debit_keywords"]:
            if keyword in description:
                if amount > 0:  # If positive, make negative
                    tx["amount"] = -abs(amount)
                    tx["normalization_applied"] = "debit_normalized_to_negative"
                break

        return tx

    def _normalize_description(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize transaction descriptions."""
        tx = transaction.copy()
        description = tx.get("description", "")

        # Apply cleaning patterns
        for pattern in self.normalization_rules["description_cleaning"]["remove_patterns"]:
            description = re.sub(pattern, ' ', description)

        # Standardize spacing
        if self.normalization_rules["description_cleaning"]["standardize_spacing"]:
            description = ' '.join(description.split())

        tx["description"] = description.strip()
        return tx

    def _apply_document_specific_formatting(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Apply document-specific formatting rules."""
        # This would implement document-type specific formatting
        # For now, return as-is
        return transaction

    async def _store_document_intelligence_in_vector_db(self, file_paths: List[str], extracted_data: List[Dict[str, Any]]):
        """Store document intelligence in vector database for semantic search."""
        try:
            for file_path in file_paths:
                # Extract and process document content
                doc_content = await self._extract_pdf_content_intelligently(file_path)

                if doc_content.get("raw_text"):
                    # Create document chunks for vector storage
                    chunks = self._create_document_chunks(doc_content)

                    # Generate embeddings for each chunk
                    texts_to_embed = [chunk["text"] for chunk in chunks]
                    embeddings = await self.embedding_generator.generate_embeddings_batch(texts_to_embed)

                    # Store in vector database
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        if embedding:
                            document_id = f"pdf_intel_{Path(file_path).stem}_{i}"

                            await self.vector_store.store_document_chunk(
                                document_id=document_id,
                                chunk_index=i,
                                content=chunk["text"],
                                embedding=embedding,
                                metadata={
                                    "source_file": file_path,
                                    "chunk_type": chunk["type"],
                                    "page_number": chunk.get("page_number"),
                                    "extraction_method": doc_content.get("metadata", {}).get("extraction_method"),
                                    "agent": "Dr. Evelyn Sharpe",
                                    "processing_timestamp": datetime.utcnow().isoformat()
                                }
                            )

                    self.logger.info(f"Stored {len(chunks)} document chunks in vector database for {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to store document intelligence in vector database: {e}")

    def _create_document_chunks(self, doc_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create intelligent document chunks for vector storage."""
        chunks = []

        # Chunk by pages
        for page in doc_content.get("pages", []):
            if page.get("text") and len(page["text"].strip()) > 50:
                chunks.append({
                    "text": page["text"],
                    "type": "page_content",
                    "page_number": page.get("page_number"),
                    "char_count": page.get("char_count", 0)
                })

        # If no page chunks, create chunks from raw text
        if not chunks and doc_content.get("raw_text"):
            raw_text = doc_content["raw_text"]
            chunk_size = 1000
            overlap = 200

            for i in range(0, len(raw_text), chunk_size - overlap):
                chunk_text = raw_text[i:i + chunk_size]
                if len(chunk_text.strip()) > 100:  # Minimum chunk size
                    chunks.append({
                        "text": chunk_text,
                        "type": "text_chunk",
                        "chunk_index": len(chunks),
                        "char_count": len(chunk_text)
                    })

        return chunks

    def _generate_vector_enhanced_forensic_analysis(self, normalized_data: List[Dict[str, Any]], processing_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive vector-enhanced forensic analysis report."""
        return {
            "summary": f"Vector-enhanced forensic analysis completed on {len(normalized_data)} transactions from {len(processing_results)} documents",
            "vector_intelligence": {
                "semantic_search_enabled": True,
                "embedding_dimension": 1536,
                "documents_stored": len(processing_results),
                "chunks_created": sum(r.get("chunks_stored", 0) for r in processing_results)
            },
            "document_analysis": {
                "ocr_applied": any(r.get("ocr_applied", False) for r in processing_results),
                "extraction_methods": list(set(r.get("extraction_method", "unknown") for r in processing_results)),
                "average_confidence": sum(r.get("quality_score", 0) for r in processing_results) / len(processing_results) if processing_results else 0
            },
            "anomaly_detection": {},
            "compliance_check": {},
            "validation_summary": {
                "total_validated": len([tx for tx in normalized_data if tx.get("forensic_validation", {}).get("valid", False)]),
                "validation_rate": len([tx for tx in normalized_data if tx.get("forensic_validation")]) / len(normalized_data) if normalized_data else 0
            }
        }

    def _generate_forensic_analysis(self, normalized_data: List[Dict[str, Any]], processing_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive forensic analysis report."""
        return {
            "summary": f"Forensic analysis completed on {len(normalized_data)} transactions",
            "document_analysis": {},
            "anomaly_detection": {},
            "compliance_check": {},
            "validation_summary": {}
        }

    def _calculate_document_intelligence_metrics(self, processing_results: List[Dict[str, Any]], normalized_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics for document intelligence processing."""
        successful_files = sum(1 for result in processing_results if result["success"])
        total_files = len(processing_results)

        return {
            "overall_quality": successful_files / total_files if total_files > 0 else 0.0,
            "extraction_accuracy": len(normalized_data) / max(1, sum(r.get("transaction_count", 0) for r in processing_results)),
            "forensic_validation_rate": 1.0,  # Would be calculated based on validation results
            "document_structure_recognition": successful_files / total_files if total_files > 0 else 0.0
        }

    def _generate_document_provenance_tracking(self, processing_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed document provenance tracking."""
        return {
            "extraction_timestamp": datetime.utcnow().isoformat(),
            "agent_version": "1.0.0",
            "processing_methodology": "Forensic PDF document intelligence with financial pattern recognition",
            "forensic_analysis_applied": True,
            "file_processing_details": processing_results
        }

    def _calculate_file_quality_score(self, transactions: List[Dict[str, Any]]) -> float:
        """Calculate quality score for a processed file."""
        if not transactions:
            return 0.0

        # Average forensic validation scores
        validation_scores = [tx.get("forensic_validation", {}).get("confidence_score", 0.5) for tx in transactions]
        avg_validation = sum(validation_scores) / len(validation_scores)

        # Completeness score
        complete_transactions = sum(1 for tx in transactions if tx.get("date") and tx.get("amount") and tx.get("description"))
        completeness = complete_transactions / len(transactions)

        return (avg_validation + completeness) / 2
