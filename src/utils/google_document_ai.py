"""
Google Document AI integration for financial document processing
"""

import base64
import json
import logging
import os
from typing import Dict, List, Any, Optional
import requests
from google.auth import default
from google.auth.transport.requests import Request
import google.auth


class GoogleDocumentAI:
    """Google Document AI client for processing financial documents."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.project_id = "285839178232"
        self.location = "us"
        self.processor_id = "8492d5cf7acfdf3"
        self.endpoint = f"https://us-documentai.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/processors/{self.processor_id}:process"
        
        # Initialize authentication
        self._setup_auth()
    
    def _setup_auth(self):
        """Setup Google Cloud authentication."""
        try:
            # Try to get default credentials
            self.credentials, self.project = default()
            self.logger.info("Google Cloud credentials loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Google Cloud credentials: {e}")
            self.credentials = None
    
    def _get_access_token(self) -> Optional[str]:
        """Get a valid access token for API requests."""
        if not self.credentials:
            return None
            
        try:
            # Refresh the token if needed
            if not self.credentials.valid:
                self.credentials.refresh(Request())
            return self.credentials.token
        except Exception as e:
            self.logger.error(f"Failed to get access token: {e}")
            return None
    
    def _encode_document(self, file_path: str) -> Optional[str]:
        """Encode document to base64."""
        try:
            with open(file_path, 'rb') as file:
                content = file.read()
            return base64.b64encode(content).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to encode document {file_path}: {e}")
            return None
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document using Google Document AI.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Processed document data with extracted entities and text
        """
        self.logger.info(f"Processing document with Google Document AI: {file_path}")
        
        # Get access token
        access_token = self._get_access_token()
        if not access_token:
            return {"error": "Failed to get access token", "transactions": []}
        
        # Encode document
        encoded_content = self._encode_document(file_path)
        if not encoded_content:
            return {"error": "Failed to encode document", "transactions": []}
        
        # Prepare request
        request_body = {
            "inlineDocument": {
                "mimeType": "application/pdf",
                "content": encoded_content
            }
        }
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=utf-8"
        }
        
        try:
            # Make API request
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=request_body,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return self._extract_financial_data(result, file_path)
            else:
                error_msg = f"Document AI API error: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return {"error": error_msg, "transactions": []}
                
        except Exception as e:
            error_msg = f"Error calling Document AI API: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg, "transactions": []}
    
    def _extract_financial_data(self, api_response: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """
        Extract financial transactions from Document AI response.
        
        Args:
            api_response: Response from Document AI API
            file_path: Original file path for reference
            
        Returns:
            Extracted financial data including transactions
        """
        try:
            document = api_response.get("document", {})
            
            # Extract full text
            full_text = document.get("text", "")
            
            # Extract entities (if any)
            entities = document.get("entities", [])
            
            # Extract form fields (if any)
            pages = document.get("pages", [])
            
            # Process the extracted data to find transactions
            transactions = self._parse_transactions_from_document_ai(full_text, entities, pages)
            
            self.logger.info(f"Google Document AI extracted {len(transactions)} transactions from {file_path}")
            
            return {
                "success": True,
                "file_path": file_path,
                "full_text": full_text,
                "text_length": len(full_text),
                "entities": entities,
                "transactions": transactions,
                "raw_response": api_response
            }
            
        except Exception as e:
            error_msg = f"Error extracting financial data: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg, "transactions": []}
    
    def _parse_transactions_from_document_ai(self, text: str, entities: List[Dict], pages: List[Dict]) -> List[Dict[str, Any]]:
        """
        Parse transactions from Document AI extracted data.
        
        Args:
            text: Full extracted text
            entities: Extracted entities
            pages: Page-level data
            
        Returns:
            List of parsed transactions
        """
        transactions = []
        
        try:
            # Look for financial patterns in the extracted text
            # This is much more reliable than our previous regex approach
            lines = text.split('\n')
            
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Look for transaction patterns in clean text from Document AI
                transaction = self._parse_transaction_line(line, line_idx)
                if transaction:
                    transactions.append(transaction)
            
            # Also check entities for financial data
            for entity in entities:
                entity_transaction = self._parse_entity_transaction(entity)
                if entity_transaction:
                    transactions.append(entity_transaction)
            
            # Remove duplicates based on date and amount
            transactions = self._deduplicate_transactions(transactions)
            
        except Exception as e:
            self.logger.error(f"Error parsing transactions from Document AI data: {e}")
        
        return transactions
    
    def _parse_transaction_line(self, line: str, line_idx: int) -> Optional[Dict[str, Any]]:
        """Parse a single line for transaction data."""
        import re
        
        # Enhanced patterns for clean Document AI text
        patterns = [
            # Date Amount Description format: "01/05 11,000.00 DEPOSIT 7676050354"
            r'(\d{1,2}/\d{1,2})\s+([\d,]+\.?\d*)\s+(DEPOSIT|WITHDRAWAL|INTEREST|TRANSFER|CHECK|PAYMENT)\s*(.*)',
            # Processed format: "Processed 01/05/24 $11000.00"
            r'Processed\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+\$?([\d,]+\.?\d*)',
            # Check format: "Check #1234 01/05 $1,500.00 Description"
            r'Check\s*#?(\d+)\s+(\d{1,2}/\d{1,2})\s+\$?([\d,]+\.?\d*)\s*(.*)',
            # ACH/Transfer format: "ACH CREDIT 01/05 $5,000.00 Description"
            r'(ACH|TRANSFER|WIRE)\s+(CREDIT|DEBIT)?\s*(\d{1,2}/\d{1,2})\s+\$?([\d,]+\.?\d*)\s*(.*)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return self._create_transaction_from_groups(match.groups(), line, line_idx)
        
        return None
    
    def _parse_entity_transaction(self, entity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse transaction data from Document AI entities."""
        # This would handle structured entities if the processor returns them
        # For now, we'll focus on text-based extraction
        return None
    
    def _create_transaction_from_groups(self, groups: tuple, original_line: str, line_idx: int) -> Dict[str, Any]:
        """Create transaction object from regex groups."""
        try:
            # Handle different group patterns
            if len(groups) >= 3:
                date = groups[0] if '/' in groups[0] else groups[1]
                amount_str = groups[1] if '/' not in groups[1] else groups[2]
                description = ' '.join(groups[2:]) if len(groups) > 2 else "Transaction"
                
                # Clean amount
                amount = float(amount_str.replace(',', '').replace('$', ''))
                
                return {
                    "date": date,
                    "amount": amount,
                    "description": description.strip(),
                    "original_line": original_line,
                    "line_index": line_idx,
                    "source": "google_document_ai",
                    "transaction_type": self._determine_transaction_type(description)
                }
        except Exception as e:
            self.logger.warning(f"Error creating transaction from groups {groups}: {e}")
        
        return None
    
    def _determine_transaction_type(self, description: str) -> str:
        """Determine transaction type from description."""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ['deposit', 'credit', 'payment received']):
            return 'credit'
        elif any(word in description_lower for word in ['withdrawal', 'debit', 'payment', 'check']):
            return 'debit'
        elif 'transfer' in description_lower:
            return 'transfer'
        elif 'interest' in description_lower:
            return 'interest'
        else:
            return 'other'
    
    def _deduplicate_transactions(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate transactions based on date and amount."""
        seen = set()
        unique_transactions = []
        
        for transaction in transactions:
            key = (transaction.get('date'), transaction.get('amount'))
            if key not in seen:
                seen.add(key)
                unique_transactions.append(transaction)
        
        return unique_transactions
