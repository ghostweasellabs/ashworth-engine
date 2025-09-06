#!/usr/bin/env python3
"""
Test Gemma3 direct PDF analysis
"""

import asyncio
from src.agents.pdf_document_intelligence import PDFDocumentIntelligenceAgent

async def test_gemma3_pdf():
    print('🤖 Testing Gemma3 direct PDF analysis...')
    agent = PDFDocumentIntelligenceAgent()
    
    file_path = 'uploads/STATEMENT-XXXXXX2223-2024-01-29.pdf'
    
    try:
        # Test direct PDF analysis with Gemma3
        result = await agent._extract_content_with_gemma3_vision(file_path)
        
        print(f"📊 Gemma3 Result:")
        print(f"  Raw text length: {len(result.get('raw_text', ''))}")
        print(f"  Pages: {len(result.get('pages', []))}")
        print(f"  Method: {result.get('metadata', {}).get('extraction_method', 'Unknown')}")
        
        if result.get('raw_text'):
            print(f"\n📝 First 500 characters:")
            print(result['raw_text'][:500])
            print("...")
            
            # Look for financial patterns
            text = result['raw_text']
            if any(word in text.lower() for word in ['deposit', 'withdrawal', 'balance', 'transaction', 'payment']):
                print("✅ Found financial content!")
            else:
                print("❌ No obvious financial content detected")
        else:
            print("❌ No text extracted")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_gemma3_pdf())
