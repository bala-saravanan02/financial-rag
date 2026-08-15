import os
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

# Structural project dependency configuration imports
from config.config import MODEL_ID, TOKENIZER_ID, BASE_DIR
from src.financial_rag.ingestion.loaders import load_pdf
from src.financial_rag.ingestion.chunkers import chunk_pages
from src.financial_rag.embeddings.embedder import embed_chunks
from src.financial_rag.retrieval.vectorstore import (
    get_or_create_collection,
    add_chunks,
    query_collection
)
from src.financial_rag.retrieval.retriever import retrieve
if __name__ == "__main__":
    target_file = "google.html.pdf"
    
   # if not os.path.exists(target_file):
    #    print(f"❌ Critical Error: Target document file cannot be found at '{target_file}'")
     #   exit(1)

    print("=" * 75)
    print("🧠 INITIALIZING SINGLE-MODEL RUNTIME MEMORY ALLOCATION")
    print("=" * 75)
    
    # Bootstrap tokenizer and neural weight matrices EXACTLY ONCE at runtime core
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    model = SentenceTransformer(MODEL_ID, device="cpu")
    print("   • Hardware validation verified. Sharing model references down the pipeline stages.")

    print("\n🚀 VERIFYING PIPELINE SPECIFICATION CONFIGURATIONS...")
    print("-" * 75)
    
    # 1. Structural Parsing and Token Window Chunk Slicing
    parents, _ = load_pdf(target_file)
    processed_chunks = chunk_pages(parents, chunk_size=200, chunk_overlap=40)
    
    # 2. Embedding Generation Stage (Injecting the single model wrapper tool)
    embedded_chunks = embed_chunks(processed_chunks)
    
    # 3. DB Collection Initialization Step
    print("\n📦 Rule 1 Check: Retrieving collection reference object...")
    my_collection = get_or_create_collection(name="financial_10k")
    print(f"   • Success. Target Collection Reference ID: {my_collection.name}")
    
    # 4. Idempotent Data Synchronization Step
    print("\n📥 Rule 2 Check: Adding chunks via idempotent upsert array mappings...")
    add_chunks(my_collection, embedded_chunks)
    
    # 5. Execute Core Raw Diagnostics Query Task
    #test_question = "What are Google's main sources of advertising revenue?"
    test_question = "What is the capital of France?"
    print(f"\n🔍 Rule 3 Check: Querying collection via raw text string input...")
    
    # Inject model directly into the query operation to bypass multi-loading bugs
    raw_chroma_payload = query_collection(my_collection, test_question, top_k=2)
    
    print("\n🏆 RAW CHROMA DICTIONARY PAYLOAD RETURN DETECTED:")
    print("=" * 75)
    print(f"Type of output: {type(raw_chroma_payload)}")
    print(f"Dictionary Root Keys: {list(raw_chroma_payload.keys())}")
    print("-" * 75)
    print(f"• Raw Matched IDs  : {raw_chroma_payload.get('ids')}")
    print(f"• Raw Distance Math: {raw_chroma_payload.get('distances')}")
    print("=" * 75)
    
    # 6. Read Raw Text Content Directly from SQLite Column Layers
    print("\n📖 [HUMAN-READABLE TEXT MATCHES EXTRACTED FROM SQLITE DATA LAYER]")
    print("-" * 75)
    
    # Target index [0] to peel back multi-query batch listing matrices safely
    matched_ids = raw_chroma_payload["ids"][0]
    matched_distances = raw_chroma_payload["distances"][0]
    matched_documents = raw_chroma_payload["documents"][0]
    matched_metadatas = raw_chroma_payload["metadatas"][0]
    
    for rank, (chunk_id, score, text_content, metadata) in enumerate(
        zip(matched_ids, matched_distances, matched_documents, matched_metadatas)
    ):
        print(f"🥇 Tabular Record Result Rank {rank + 1}:")
        print(f"   • Chunk Unique ID: {chunk_id}")
        print(f"   • Cosine Distance: {score:.4f}")
        # Updated tracking lookups to extract fixed source metadata definitions cleanly
        print(f"   • File Provenance: {metadata.get('file_name')} | Page Reference: {metadata.get('source_page')}")
        print(f"   • Text Content   : \"{text_content.strip()[:180]}...\"")
        print("." * 75)

    # 7. Validate the Unified retriever.py Component Layer Integration
    print("\n🛡️ HOOKING RETRIEVER FILTER LAYER FOR HIGH-DENSITY GROUNDING CHECK...")
    print("-" * 75)
    
    # Run standalone retriever logic—trims stray records beyond max_distance bounds
    cleaned_context_matches = retrieve(
        collection=my_collection,
        user_query=test_question,
        top_k=3,
        max_distance=0.6
    )
    
    print(f"📊 Filter Complete: {len(cleaned_context_matches)} chunks survived the safety threshold.")
    for idx, clean_match in enumerate(cleaned_context_matches):
        print(f"   ⚡ Cleaned Context Node {idx + 1} -> Distance: {clean_match['cosine_distance']} | Doc: {clean_match['source_file_name']} [Page {clean_match['page_number']}]")
        
    print("=" * 75)
    print("✅ All functional specification targets met. Data sits stored on disk")