import os
from typing import List, Dict, Any
import chromadb
from config.config import BASE_DIR,MODEL_ID
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(MODEL_ID, device="cpu")
DB_PATH = BASE_DIR / "data" / "chroma_db"

os.makedirs(DB_PATH, exist_ok=True)
_client = chromadb.PersistentClient(path = DB_PATH)

def get_or_create_collection(name: str):

    return _client.get_or_create_collection(name = name, metadata ={"hnsw:space":"cosine"})


def add_chunks(collection, chunks: List[dict[str,Any]]) -> None:

    if not chunks:
        raise ValueError("Ingestion array is empty . Skipping the storage cycle")

    ids: List[str] = []
    embeddings: List[str] = []
    documents: List[str] = []
    metadata: List[dict[str,Any]] = []

    for chunk in chunks:
            ids.append(chunk["chunk_id"])
            embeddings.append(chunk["embedding"])
            documents.append(chunk["chunk_content"])

            raw_metadata = {
                 "parent_id":chunk["parent_id"],
                 "file_name":chunk["source_file_name"],
                 "source_page":chunk["page_number"]
                 
            }
            sanitized_metadata = {
            k: v for k, v in raw_metadata.items()
            if v is not None and isinstance(v, (str, int, float, bool))
            }
            metadata.append(sanitized_metadata)

    collection.upsert(ids = ids, embeddings = embeddings, documents = documents, metadatas = metadata)
    print("stored in chroma db success")
    print(f"Total chunks embedded this run: {len(chunks)}") # Fixed variable
    print(f"Total items now in collection: {collection.count()}")

def query_collection(collection, user_query: str, top_k: int = 5) -> dict:

    embeddings = model.encode(inputs = user_query, normalize_embeddings = True).tolist()
    return collection.query(
        query_embeddings=[embeddings],
        n_results=top_k
    )

