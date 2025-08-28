"""
Image file processor with OCR capabilities.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

# Optional imports with fallbacks
try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    logging.warning("OCR libraries not available, image processing disabled")

from .base import BaseFileProcessor
from .exceptions import DataExtractionError
from .pdf import PDFProcessor

logger = logging.getLogger(__name__)


class ImageProcessor(BaseFileProcessor):
    """Processor for image files with OCR."""
    
    def process(self, file_path: Path, source_name: str) -> List[Dict[str, Any]]:
        """Process image file using OCR and text parsing."""
        if not HAS_OCR:
            raise DataExtractionError("OCR libraries not available for image processing")
        
        transactions = []
        
        try:
            # Extract text using OCR
            text = self._extract_text_from_image(file_path)
            
            if not text.strip():
                logger.warning(f"No text extracted from image {file_path}")
                return transactions
            
            # Use PDF processor's text parsing logic
            pdf_processor = PDFProcessor()
            transactions = pdf_processor._parse_transactions_from_text(text, source_name)
            
        except Exception as e:
            logger.error(f"Failed to process image file {file_path}: {e}")
            raise DataExtractionError(f"Cannot process image file: {e}")
        
        return transactions
    
    def _extract_text_from_image(self, file_path: Path) -> str:
        """Extract text from image using OCR."""
        try:
            # Open and process image
            image = Image.open(file_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(image)
            
            logger.info(f"Extracted {len(text)} characters from image {file_path}")
            return text
            
        except Exception as e:
            logger.error(f"Failed to extract text from image: {e}")
            raise