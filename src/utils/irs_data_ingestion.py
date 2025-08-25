"""Specialized data ingestion utilities for IRS documents and financial regulations"""

import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.utils.document_ingestion import get_document_ingestion
from src.utils.memory_store import get_shared_memory_store, MemoryNamespaces
from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

class IRSDataIngestion:
    """Specialized ingestion for IRS documents with tax-specific metadata extraction"""
    
    def __init__(self):
        self.document_ingestion = get_document_ingestion()
        self.memory_store = get_shared_memory_store()
        
        # IRS document patterns for classification
        self.irs_patterns = {
            "publication": re.compile(r"Publication\s+(\d+)", re.IGNORECASE),
            "form": re.compile(r"Form\s+(\d+[\w-]*)", re.IGNORECASE),
            "instruction": re.compile(r"Instructions?\s+for\s+Form\s+(\d+[\w-]*)", re.IGNORECASE),
            "revenue_ruling": re.compile(r"Revenue\s+Ruling\s+(\d{4}-\d+)", re.IGNORECASE),
            "revenue_procedure": re.compile(r"Revenue\s+Procedure\s+(\d{4}-\d+)", re.IGNORECASE),
            "notice": re.compile(r"Notice\s+(\d{4}-\d+)", re.IGNORECASE),
            "circular": re.compile(r"Circular\s+(\w+)", re.IGNORECASE)
        }
        
        # Tax year pattern
        self.tax_year_pattern = re.compile(r"(?:tax\s+year\s+|for\s+)(\d{4})", re.IGNORECASE)
        
        # Business expense categories from IRS Publication 334
        self.expense_categories = {
            "advertising": ["advertising", "marketing", "promotion"],
            "car_and_truck": ["vehicle", "car", "truck", "transportation", "mileage"],
            "commissions_fees": ["commission", "professional fee", "consultant"],
            "contract_labor": ["contractor", "freelancer", "1099"],
            "depletion": ["depletion", "natural resources"],
            "depreciation": ["depreciation", "equipment", "machinery"],
            "employee_benefits": ["health insurance", "retirement", "benefits"],
            "insurance": ["insurance", "liability", "property"],
            "interest": ["loan interest", "business interest"],
            "legal_professional": ["legal", "attorney", "CPA", "accounting"],
            "office_expense": ["supplies", "postage", "stationery"],
            "pension_profit_sharing": ["pension", "profit sharing", "401k"],
            "rent_lease": ["rent", "lease", "office space"],
            "repairs_maintenance": ["repair", "maintenance", "upkeep"],
            "supplies": ["materials", "inventory", "supplies"],
            "taxes_licenses": ["business tax", "license", "permit"],
            "travel": ["travel", "lodging", "meals"],
            "meals_entertainment": ["meals", "entertainment", "business dining"],
            "utilities": ["utilities", "phone", "internet", "electricity"],
            "wages": ["salary", "wages", "payroll"],
            "other": ["miscellaneous", "other"]
        }
    
    def _extract_irs_metadata(self, content: str, file_name: str) -> Dict[str, Any]:
        """Extract IRS-specific metadata from document content"""
        metadata = {
            "document_type": "unknown",
            "document_number": None,
            "tax_year": None,
            "expense_categories": [],
            "keywords": [],
            "compliance_level": "standard"
        }
        
        # Classify document type
        for doc_type, pattern in self.irs_patterns.items():
            match = pattern.search(content)
            if match:
                metadata["document_type"] = doc_type
                metadata["document_number"] = match.group(1)
                break
        
        # Extract tax year
        tax_year_match = self.tax_year_pattern.search(content)
        if tax_year_match:
            metadata["tax_year"] = int(tax_year_match.group(1))
        
        # Identify relevant expense categories
        content_lower = content.lower()
        for category, keywords in self.expense_categories.items():
            if any(keyword in content_lower for keyword in keywords):
                metadata["expense_categories"].append(category)
        
        # Extract key terms for search optimization
        key_terms = []
        if "deductible" in content_lower:
            key_terms.append("deductible")
        if "section 179" in content_lower:
            key_terms.append("section_179")
        if "business meal" in content_lower:
            key_terms.append("business_meals")
        if "form 8300" in content_lower:
            key_terms.append("form_8300")
        
        metadata["keywords"] = key_terms
        
        # Set compliance level for critical documents
        if any(term in content_lower for term in ["must", "required", "shall", "penalty"]):
            metadata["compliance_level"] = "critical"
        elif any(term in content_lower for term in ["should", "recommended", "may"]):
            metadata["compliance_level"] = "recommended"
        
        return metadata
    
    async def ingest_irs_publication(self, 
                                   file_path: str,
                                   publication_number: Optional[str] = None,
                                   tax_year: Optional[int] = None) -> List[str]:
        """Ingest IRS publication with specialized metadata"""
        try:
            # Read file content for metadata extraction
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            irs_metadata = self._extract_irs_metadata(content, file_path)
            
            # Add provided metadata
            if publication_number:
                irs_metadata["publication_number"] = publication_number
            if tax_year:
                irs_metadata["tax_year"] = tax_year
            
            # Additional IRS-specific metadata
            additional_metadata = {
                **irs_metadata,
                "source": "irs_publication",
                "authority_level": "official",
                "compliance_source": True,
                "last_updated": datetime.now().isoformat(),
                "reference_priority": "high"
            }
            
            # Ingest into IRS documents collection
            doc_ids = await self.document_ingestion.ingest_file(
                file_path=file_path,
                collection_name="irs_documents",
                namespace="publications",
                additional_metadata=additional_metadata
            )
            
            # Store in shared memory for quick access
            await self._store_irs_reference(irs_metadata, doc_ids)
            
            logger.info(f"Successfully ingested IRS publication: {file_path}")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to ingest IRS publication {file_path}: {e}")
            raise
    
    async def ingest_tax_guidance(self, 
                                file_path: str,
                                guidance_type: str = "general",
                                subject_areas: Optional[List[str]] = None) -> List[str]:
        """Ingest tax guidance documents"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            irs_metadata = self._extract_irs_metadata(content, file_path)
            
            additional_metadata = {
                **irs_metadata,
                "source": "tax_guidance",
                "guidance_type": guidance_type,
                "subject_areas": subject_areas or [],
                "authority_level": "guidance",
                "compliance_source": True,
                "reference_priority": "medium"
            }
            
            doc_ids = await self.document_ingestion.ingest_file(
                file_path=file_path,
                collection_name="tax_guidance",
                namespace=guidance_type,
                additional_metadata=additional_metadata
            )
            
            await self._store_tax_guidance_reference(irs_metadata, doc_ids, guidance_type)
            
            logger.info(f"Successfully ingested tax guidance: {file_path}")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to ingest tax guidance {file_path}: {e}")
            raise
    
    async def ingest_financial_regulations(self, 
                                         file_path: str,
                                         regulation_type: str = "general",
                                         regulatory_body: str = "unknown") -> List[str]:
        """Ingest financial regulations from various regulatory bodies"""
        try:
            additional_metadata = {
                "source": "financial_regulations",
                "regulation_type": regulation_type,
                "regulatory_body": regulatory_body,
                "authority_level": "regulatory",
                "compliance_source": True,
                "reference_priority": "high"
            }
            
            doc_ids = await self.document_ingestion.ingest_file(
                file_path=file_path,
                collection_name="financial_regulations",
                namespace=regulation_type,
                additional_metadata=additional_metadata
            )
            
            # Store regulatory reference
            await self._store_regulatory_reference(regulation_type, regulatory_body, doc_ids)
            
            logger.info(f"Successfully ingested financial regulation: {file_path}")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to ingest financial regulation {file_path}: {e}")
            raise
    
    async def _store_irs_reference(self, metadata: Dict[str, Any], doc_ids: List[str]):
        """Store IRS document reference in shared memory"""
        try:
            namespace = MemoryNamespaces.IRS_GUIDANCE
            
            reference = {
                "document_type": metadata.get("document_type"),
                "document_number": metadata.get("document_number"),
                "tax_year": metadata.get("tax_year"),
                "expense_categories": metadata.get("expense_categories", []),
                "keywords": metadata.get("keywords", []),
                "compliance_level": metadata.get("compliance_level"),
                "document_ids": doc_ids,
                "indexed_at": datetime.now().isoformat()
            }
            
            key = f"irs_{metadata.get('document_type', 'unknown')}_{metadata.get('document_number', 'unknown')}"
            await self.memory_store.put_memory(namespace, key, reference)
            
        except Exception as e:
            logger.error(f"Failed to store IRS reference: {e}")
    
    async def _store_tax_guidance_reference(self, metadata: Dict[str, Any], doc_ids: List[str], guidance_type: str):
        """Store tax guidance reference in shared memory"""
        try:
            namespace = MemoryNamespaces.IRS_GUIDANCE + ("guidance",)
            
            reference = {
                "guidance_type": guidance_type,
                "expense_categories": metadata.get("expense_categories", []),
                "keywords": metadata.get("keywords", []),
                "document_ids": doc_ids,
                "indexed_at": datetime.now().isoformat()
            }
            
            key = f"guidance_{guidance_type}_{len(doc_ids)}"
            await self.memory_store.put_memory(namespace, key, reference)
            
        except Exception as e:
            logger.error(f"Failed to store tax guidance reference: {e}")
    
    async def _store_regulatory_reference(self, regulation_type: str, regulatory_body: str, doc_ids: List[str]):
        """Store regulatory document reference in shared memory"""
        try:
            namespace = MemoryNamespaces.FINANCIAL_RULES
            
            reference = {
                "regulation_type": regulation_type,
                "regulatory_body": regulatory_body,
                "document_ids": doc_ids,
                "indexed_at": datetime.now().isoformat()
            }
            
            key = f"regulation_{regulatory_body}_{regulation_type}"
            await self.memory_store.put_memory(namespace, key, reference)
            
        except Exception as e:
            logger.error(f"Failed to store regulatory reference: {e}")
    
    async def setup_default_irs_knowledge(self):
        """Setup default IRS knowledge base with key publications"""
        try:
            # Create essential IRS knowledge entries
            essential_knowledge = [
                {
                    "content": "Business expenses must be both ordinary and necessary. An ordinary expense is one that is common and accepted in your trade or business. A necessary expense is one that is helpful and appropriate for your trade or business.",
                    "metadata": {
                        "source": "irs_publication_334",
                        "document_type": "publication",
                        "expense_categories": ["general"],
                        "compliance_level": "critical",
                        "keywords": ["ordinary", "necessary", "business_expense"]
                    }
                },
                {
                    "content": "Generally, you can deduct only 50% of your business meal expenses. However, meals provided for the convenience of the employer on the business premises are 100% deductible.",
                    "metadata": {
                        "source": "irs_publication_334",
                        "document_type": "publication",
                        "expense_categories": ["meals_entertainment"],
                        "compliance_level": "critical",
                        "keywords": ["business_meals", "50_percent_rule"]
                    }
                },
                {
                    "content": "Section 179 allows you to deduct the full purchase price of qualifying equipment and software purchased or financed during the tax year. For 2025, the maximum Section 179 expense deduction is $2.5 million.",
                    "metadata": {
                        "source": "irs_section_179",
                        "document_type": "tax_code",
                        "expense_categories": ["depreciation"],
                        "compliance_level": "critical",
                        "keywords": ["section_179", "equipment_deduction"]
                    }
                },
                {
                    "content": "If you receive more than $10,000 in cash in your trade or business, you must file Form 8300. This includes a series of related transactions totaling more than $10,000 within a 12-month period.",
                    "metadata": {
                        "source": "irs_form_8300",
                        "document_type": "form",
                        "expense_categories": ["other"],
                        "compliance_level": "critical",
                        "keywords": ["form_8300", "cash_reporting"]
                    }
                }
            ]
            
            # Ingest essential knowledge
            doc_ids_list = []
            for knowledge in essential_knowledge:
                doc_ids = await self.document_ingestion.ingest_text(
                    text=knowledge["content"],
                    collection_name="irs_documents",
                    namespace="essential_knowledge",
                    metadata=knowledge["metadata"],
                    title=f"Essential IRS Knowledge - {knowledge['metadata']['source']}"
                )
                doc_ids_list.extend(doc_ids)
            
            logger.info(f"Setup default IRS knowledge: {len(doc_ids_list)} documents")
            return doc_ids_list
            
        except Exception as e:
            logger.error(f"Failed to setup default IRS knowledge: {e}")
            raise

# Global IRS data ingestion instance
irs_data_ingestion = IRSDataIngestion()

def get_irs_data_ingestion() -> IRSDataIngestion:
    """Get the global IRS data ingestion instance"""
    return irs_data_ingestion