from typing import Any,List,Dict
from src.financial_rag.retrieval.vectorstore import query_collection

def retrieve(collection, user_query: str, top_k: int=5, max_distance: float=0.6) -> list[dict[str,Any]]:

    raw_chroma_payload = query_collection(collection,user_query,top_k)

    if not raw_chroma_payload or not raw_chroma_payload.get("ids") or not raw_chroma_payload["ids"][0]:
        return []

    matched_ids = raw_chroma_payload["ids"][0]
    matched_distances = raw_chroma_payload["distances"][0]
    matched_documents = raw_chroma_payload["documents"][0]
    matched_metadatas = raw_chroma_payload["metadatas"][0]

    formatted_results=[]
    for chunk_id, distance_score, text_content, metadata in zip(
        matched_ids, matched_distances, matched_documents, matched_metadatas
    ):
        # 🛡️ THE MAX DISTANCE THRESHOLD FILTER
        # If the score is higher than our limit, it's too unrelated—discard it!
        if distance_score > max_distance:
            continue

        # Build a safe, cleaned-up output object mapping back metadata correctly
        cleaned_result = {
            "chunk_id": chunk_id,
            "cosine_distance": round(float(distance_score), 4),
            "source_file_name": metadata.get("file_name", "unknown_document"),
            "page_number": metadata.get("source_page", "unknown_page"),
            "chunk_content": text_content.strip()
        }
        
        formatted_results.append(cleaned_result)

    return formatted_results

