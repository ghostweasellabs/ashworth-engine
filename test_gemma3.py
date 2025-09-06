#!/usr/bin/env python3
"""
Test Gemma3 vision capabilities directly
"""

import asyncio
import base64
from pdf2image import convert_from_path
from src.agents.pdf_document_intelligence import PDFDocumentIntelligenceAgent

async def test_gemma3_vision():
    """Test Gemma3 vision with a real PDF page"""

    print("🔍 Testing Gemma3 Vision Capabilities...")

    # Convert first page of a PDF to image
    try:
        print("📄 Converting PDF to image...")
        images = convert_from_path('uploads/STATEMENT-XXXXXX2223-2024-01-29.pdf', first_page=1, last_page=1)

        if not images:
            print("❌ No images generated from PDF")
            return

        # Convert to base64
        from io import BytesIO
        buffered = BytesIO()
        images[0].save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        print(f"📷 Created base64 image ({len(img_base64)} chars)")

        # Test Gemma3 vision
        agent = PDFDocumentIntelligenceAgent()

        print("🤖 Calling Gemma3 vision analysis...")
        result = await agent._analyze_image_with_gemma3(img_base64, 1)

        print("📊 Results:")
        print(f"   Text extracted: {len(result.get('text', ''))} characters")
        print(f"   Confidence: {result.get('confidence', 0):.2f}")
        print(f"   Entities found: {result.get('extracted_entities', {})}")

        if result.get('text'):
            print("\n📝 Extracted Text (first 500 chars):")
            print(result['text'][:500])

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_gemma3_vision())
