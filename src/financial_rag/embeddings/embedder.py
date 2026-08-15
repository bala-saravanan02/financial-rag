from sentence_transformers import SentenceTransformer
from config.config import MODEL_ID


model = SentenceTransformer(MODEL_ID, device="cpu")

def embed_chunks(chunks: list[dict]) -> list[dict]:

    if not chunks:
        return []

    chunk_content_to_embed  = [chunk.get("chunk_content") for chunk in chunks]

    embeddings = model.encode(inputs = chunk_content_to_embed,
                              batch_size = 128, normalize_embeddings=True,show_progress_bar=True)

    for idx , embedding_vector in enumerate(embeddings):
        chunks[idx]["embedding"] = embedding_vector.tolist()

    return chunks


