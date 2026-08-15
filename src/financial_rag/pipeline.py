import os
import random
from src.financial_rag.ingestion.loaders import load_pdf
from src.financial_rag.ingestion.chunkers import chunk_pages
from transformers import AutoTokenizer

TOKENIZER_ID = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

if __name__ == "__main__":
    target_file = "google.html.pdf"
    if not os.path.exists(target_file) and os.path.exists(f"data/{target_file}"):
        target_file = f"data/{target_file}"
        
    print(f"🔄 Processing '{target_file}' for structural inspection...\n")
    parents, children = load_pdf(target_file)
    
    print("=" * 60)
    print("🔍 VISUAL INSPECTION MODE (RANDOM SAMPLE + CHUNKING)")
    print("=" * 60)
    
    # 1. Inspect a RANDOM Parent Page text block & process it through the chunker
    if parents:
        random_parent_idx = random.randint(0, len(parents) - 1)
        sample_parent = parents[random_parent_idx]
        
        print(f"\n📄 [SAMPLE PARENT PAGE CHUNK (Random Index: {random_parent_idx})]")
        print("-" * 40)
        
        text_content = sample_parent.get('page_content', '') if isinstance(sample_parent, dict) else (
            sample_parent.page_content if hasattr(sample_parent, 'page_content') else str(sample_parent)
        )
        
        print("--- [Original Text Sample (First 300 Chars)] ---")
        print(text_content[:300] + "\n...")
        
        # Run the safe chunking execution pipeline
        processed_chunks = chunk_pages(parents, chunk_size=200, chunk_overlap=40)
        
        print(f"\n📦 [Chunking Results: Generated {len(processed_chunks)} sub-chunks]")

    for i, chunk in enumerate(processed_chunks[:3]): # Preview first 3 chunks max
        # 1. Extract the raw text string from the dictionary format
        text_to_encode = chunk.get("chunk_content", "")
    
        # 2. Safely encode the text string
        chunk_tokens = len(tokenizer.encode(text_to_encode, add_special_tokens=False))
    
        # 3. Print the verified chunk token count and preview text
        print(f" 👉 Sub-chunk {i+1}: {chunk_tokens} verified tokens | Text: {text_to_encode[:100]}...")

    print("-" * 40)
        
    # 2. Inspect a RANDOM Child Table layout
    if children:
        random_child_idx = random.randint(0, len(children) - 1)
        sample_child = children[random_child_idx]
        
        print(f"\n📊 [SAMPLE CHILD TABLE (Random Index: {random_child_idx})]")
        print("-" * 40)
        table_content = sample_child.get('table_content', '') if isinstance(sample_child, dict) else (
            sample_child.page_content if hasattr(sample_child, 'page_content') else str(sample_child)
        )
        print(table_content)
        print("-" * 40)
    else:
        print("\n❌ No child tables found in the extraction list to display.")
        
    print(f"\n✅ Inspection Finished. Total Parents: {len(parents)} | Total Child Tables: {len(children)}")