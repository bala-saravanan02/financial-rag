from config.config import BASE_DIR,ALLOWED_EXTENSIONS,ALLOWED_MIME_TYPES
import pymupdf
import magic
from pathlib import Path
from typing import List
import uuid
import os
import pytesseract
from dotenv import load_dotenv
load_dotenv()

tesseract_cmd = os.getenv("TESSERACT_CMD")

# 2. Check if the file physically exists on the disk layout
if not os.path.exists(tesseract_cmd):
    raise RuntimeError(f"Tesseract executable binary not found at: {tesseract_cmd}")

# 3. Assign the valid verified path to the pytesseract wrapper package
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

def load_pdf(file_name) -> List[dict]:

    #Defining the file Path
    file_path = BASE_DIR / "data" / "raw" / file_name

    #Validation check1: Supporting the right extension
    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {file_path.suffix()}")

    #Validation check2: magic number validation
    magic_number = magic.from_file(str(file_path), mime = True)
    if magic_number not in ALLOWED_MIME_TYPES:
        raise ValueError(f"malicious file is detected! Content Type: {magic_number}")

    #Open the file
    try:
        doc = pymupdf.open(file_path)
        total_pages = doc.page_count
    except (EnvironmentError, RuntimeError) as e:
        raise ValueError(
        f"Secure Processing Blocked: Failed to open '{file_path}'. "
        f"The file may be missing, locked, or internally corrupted."
    ) from e

    document_payload = []
    parent_pages = []
    child_tables = []
    #Extraction of text from the pdf
    for page_number in range(total_pages):

        page = doc.load_page(page_number)
       # parent_id = f"{file_name}_page_{page_number + 1}_{uuid.uuid4().hex[:8]}"
        parent_id = f"{file_name}_page_{page_number+1}"
        is_ocr_triggered = False

        page_text = page.get_text("text").strip()

        if not page_text:
            page_dict = page.get_text("dict")
            has_images = any(block["type"] == 1 for block in page_dict.get("blocks", []))

            if has_images or len(page.get_images()) > 0:
                print(f"[{file_name}] Page {page_number + 1}: Scanned page found. Running OCR...")
                ocr_textpage = page.get_textpage_ocr(language = 'eng', full = True)
                page_text = ocr_textpage.extractText().strip()
                is_ocr_triggered = True
            else:
                    print(f"[{file_name}] Page {page_number + 1}: Genuinely blank page. Skipping.")
                    continue

        
        tables = page.find_tables()
        for table_index, table in enumerate(tables.tables):
            df = table.to_pandas()
            markdown_str = df.to_markdown(index=False)

            child_payload = {
                "child_id":f"{parent_id}_table_{table_index + 1}",
                "parent_id":parent_id,
                "table_content": markdown_str,
                "metadata":{
                    "type":"table_child",
                    "source_file":file_name,
                    "page_number":page_number + 1
                }
            }
            child_tables.append(child_payload)

        parent_payload = {
            "parent_id":parent_id,
            "page_content":page_text,
            "metadata":{
                "type":"page_parent",
                "source_file":file_name,
                "page_number":page_number+1,
                "total_pages":total_pages,
                "child_table_count":len(tables.tables),
                "is_ocr":is_ocr_triggered
            }
        }
        parent_pages.append(parent_payload)

    return parent_pages, child_tables



