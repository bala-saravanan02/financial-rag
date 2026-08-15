from transformers import AutoTokenizer

TOKENIZER_ID = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

def chunk_pages(pages: list[dict], chunk_size: int=200, chunk_overlap: int=40) -> list[dict]:

   
    step = chunk_size - chunk_overlap
    chunks = []
    chunk_counter = 0

    for page in pages:
        text = page.get("page_content") or ""
        encode = tokenizer(text , add_special_tokens = False, return_attention_mask = False)
        token_ids = encode["input_ids"]
        metadata = page.get("metadata") or {}
        page_number = metadata.get("page_number","unknown_page") 
        source_file = metadata.get("source_file","unknown_file_name") 
        page_chunk_index=0
        start_idx = 0
        while start_idx < len(token_ids):
            end_idx = start_idx + chunk_size
            token_window = token_ids[start_idx : end_idx]
            page_chunk_index+=1
            decoded_text = tokenizer.decode(token_window,skip_special_tokens = False)
            chunk_id = f"{source_file}_page_{page_number}_chunk_{page_chunk_index}"
            chunks.append (
                {   
                    "chunk_id": chunk_id,
                    "parent_id":page.get("parent_id"),
                    "chunk_content":decoded_text,
                    "source_file_name":source_file,
                    "page_number":page_number
                }
            )
            start_idx += step
            if end_idx >= len(token_ids):
                break
    return chunks


