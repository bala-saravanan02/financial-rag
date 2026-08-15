import os
import random
from src.financial_rag.ingestion.loaders import load_pdf
from src.financial_rag.ingestion.chunkers import chunk_pages
from src.financial_rag.embeddings.embedder import embed_chunks
from transformers import AutoTokenizer
# Import your newly written embedding function from its module
# (Adjust this import path depending on where you saved your embed_chunks function)
from sentence_transformers import SentenceTransformer

TOKENIZER_ID = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

# Initialize the embedding model for the pipeline validation step
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_ID, device="cpu")


if __name__ == "__main__":
    target_file = "google.html.pdf"
    if not os.path.exists(target_file) and os.path.exists(f"data/{target_file}"):
        target_file = f"data/{target_file}"
        
    print(f"🔄 Processing '{target_file}' for structural inspection...\n")
    parents, children = load_pdf(target_file)
    
    print("=" * 60)
    print("🔍 VISUAL INSPECTION MODE (RANDOM SAMPLE + PIPELINE VALIDATION)")
    print("=" * 60)
    
    # 1. Inspect a RANDOM Parent Page text block
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
        
        # --- PHASE 1: Run the safe chunking execution pipeline ---
        processed_chunks = chunk_pages(parents, chunk_size=200, chunk_overlap=40)
        print(f"\n📦 [Chunking Results: Generated {len(processed_chunks)} sub-chunks]")
        
        # --- PHASE 2: Run the new embedding matrix operations pipeline ---
        print("\n🧠 [Embedding Generation: Triggering model.encode()...]")
        embedded_chunks = embed_chunks(processed_chunks)
        print("✅ Embeddings calculated and appended to dictionary structures.")
        
        # --- PHASE 3: Verify and print output for validation ---
        print("\n📋 [Verifying Pipeline Transformations (Previewing First 3 Chunks)]")
        print("-" * 40)
        for i, chunk in enumerate(embedded_chunks[:3]):
            text_to_encode = chunk.get("chunk_content", "")
            
            # Recalculate token count to inspect potential 200 -> 202 round-trip drift
            chunk_tokens = len(tokenizer.encode(text_to_encode, add_special_tokens=False))
            
            # Extract the vector to prove it exists in the dictionary format
            vector_data = chunk.get("embedding", [])
            vector_length = len(vector_data)
            
            # Preview a small slice of the first 3 numbers of the 384 array
            vector_preview = vector_data[:3] if vector_length >= 3 else []
            
            print(f" 👉 Sub-chunk {i+1}:")
            print(f"    • Token Measurement : {chunk_tokens} verified tokens (Drift Check)")
            print(f"    • Vector Dimension  : Size {vector_length} list (Target: 384)")
            print(f"    • Vector Values Mock: {vector_preview}... (Normalized coordinates)")
            print(f"    • Text Snippet      : '{text_to_encode[:80]}...'")
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
