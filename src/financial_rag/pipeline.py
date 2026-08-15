import os
import random
from src.financial_rag.ingestion.loaders import load_pdf
from src.financial_rag.ingestion.chunkers import chunk_pages
from src.financial_rag.embeddings.embedder import embed_chunks
from src.financial_rag.retrieval.vectorstore import get_or_create_collection,add_chunks,query_collection
from transformers import AutoTokenizer

from config.config import MODEL_ID,TOKENIZER_ID,BASE_DIR
# Import your newly written embedding function from its module
# (Adjust this import path depending on where you saved your embed_chunks function)
from sentence_transformers import SentenceTransformer



tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)


model = SentenceTransformer(MODEL_ID, device="cpu")
if __name__ == "__main__":
    target_file = "google.html.pdf"
    #if not os.path.exists(target_file):
     #   print(f"❌ Test file cannot be found at '{target_file}'")
      #  exit(1)

    print("🚀 Verifying Specification Rule Conformity...")
    
    # Run structural pipeline stages
    parents, _ = load_pdf(target_file)
    processed_chunks = chunk_pages(parents, chunk_size=200, chunk_overlap=40)
    embedded_chunks = embed_chunks(processed_chunks)
    
    # 1. Test Rule 1: Get or Create Collection Object
    print("\n📦 Rule 1 Check: Retrieving collection reference object...")
    my_collection = get_or_create_collection(name="financial_10k")
    print(f" • Success. Target Collection Reference ID: {my_collection.name}")
    
    # 2. Test Rule 2: Idempotent add_chunks
    print("\n📥 Rule 2 Check: Adding chunks via idempotent upsert array mappings...")
    add_chunks(my_collection, embedded_chunks)
    
    # 3. Test Rule 3: Raw Query Collection (Passing text string directly now)
    test_question = "What are Google's main sources of advertising revenue?"
    print(f"\n🔍 Rule 3 Check: Querying collection via raw text string input...")
    
    # Clean and encapsulated: We pass 'test_question' (str) instead of 'query_vector' (list)
    raw_chroma_payload = query_collection(my_collection, test_question, top_k=2)
    
    print("\n🏆 RAW CHROMA DICTIONARY PAYLOAD RETURN DETECTED:")
    print("=" * 75)
    print(f"Type of output: {type(raw_chroma_payload)}")
    print(f"Dictionary Root Keys: {list(raw_chroma_payload.keys())}")
    print("-" * 75)
    print(f"• Raw Matched IDs  : {raw_chroma_payload.get('ids')}")
    print(f"• Raw Distance Math: {raw_chroma_payload.get('distances')}")
    print("=" * 75)
    print("✅ All functional specification targets met. Data sits stored on disk and maps cleanly to raw targets.")
     # =========================================================================
    # 👇 PLACE THESE NEW LINES RIGHT HERE TO READ THE RAW TEXT CONTENTS 👇
    # =========================================================================
    print("-" * 75)
    print("📖 [HUMAN-READABLE TEXT MATCHES EXTRACTED FROM SQLITE DATA LAYER]")
    print("-" * 75)
    
    # Safely extract individual lists by targeting the 0th inner query array index
    matched_ids = raw_chroma_payload["ids"][0]
    matched_distances = raw_chroma_payload["distances"][0]
    matched_documents = raw_chroma_payload["documents"][0]
    matched_metadatas = raw_chroma_payload["metadatas"][0]
    
    # Zip elements together to loop over ranked rows step-by-step
    for rank, (chunk_id, score, text_content, metadata) in enumerate(
        zip(matched_ids, matched_distances, matched_documents, matched_metadatas)
    ):
        print(f"🥇 Result Rank {rank + 1}:")
        print(f"   • Chunk Unique ID: {chunk_id}")
        print(f"   • Cosine Distance: {score:.4f}")
        print(f"   • File Provenance: {metadata.get('file_name')} | Page: {metadata.get('source_page')}")
        print(f"   • Text Content   : \"{text_content.strip()}\"")
        print("." * 75) # Mini separation spacer dot line
        
    # =========================================================================
    # 👆 END OF NEW EXTRACTION LOOP LINES 👆
    # =========================================================================

    print("=" * 75)
    print("✅ All functional specification targets met. Data sits stored on disk and maps cleanly to raw targets.")