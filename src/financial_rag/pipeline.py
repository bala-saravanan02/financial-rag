import os
import random  # <--- Added for shuffling selection
from src.financial_rag.ingestion.loaders import load_pdf

if __name__ == "__main__":
    target_file = "google.html.pdf"
    
    if not os.path.exists(target_file) and os.path.exists(f"data/{target_file}"):
        target_file = f"data/{target_file}"
        
    print(f"🔄 Processing '{target_file}' for structural inspection...\n")
    parents, children = load_pdf(target_file)
    
    print("=" * 60)
    print("🔍 VISUAL INSPECTION MODE (RANDOM SAMPLE)")
    print("=" * 60)
    
    # 1. Inspect a RANDOM Parent Page text block instead of the first one
    if parents:
        # Pick a completely random index between 0 and the max available length
        random_parent_idx = random.randint(0, len(parents) - 1)
        sample_parent = parents[random_parent_idx]
        
        print(f"\n📄 [SAMPLE PARENT PAGE CHUNK (Random Index: {random_parent_idx})]")
        print("-" * 40)
        
        # Pull text regardless of whether it's a dict or an object instance
        text_content = sample_parent.get('page_content', '') if isinstance(sample_parent, dict) else (
            sample_parent.page_content if hasattr(sample_parent, 'page_content') else str(sample_parent)
        )
        print(text_content[:500] + "\n... [Truncated for preview] ...")
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
