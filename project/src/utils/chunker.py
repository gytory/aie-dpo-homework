from typing import List, Optional
import pandas as pd

def chunk_text(text: str, chunk_size: int = 40, overlap: int = 20) -> List[str]:
    words = text.replace("\n", " ").split()
    
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть положительным")
    if overlap >= chunk_size:
        raise ValueError("overlap должен быть меньше chunk_size")
    
    chunks = []
    step = chunk_size - overlap
    
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        
        if start + chunk_size >= len(words):
            break
    
    return chunks

def prepare_chunks(
    df_docs: pd.DataFrame,
    chunk_size: Optional[int] = 40,
    overlap: int = 20
) -> pd.DataFrame:
    rows = []
    
    for _, row in df_docs.iterrows():
        doc_id = row['doc_id']
        topic = row['topic']
        text = row['response_text']
        
        if chunk_size is None:
            rows.append({
                'chunk_id': doc_id,
                'doc_id': doc_id,
                'topic': topic,
                'text': text
            })
        else:
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for i, chunk in enumerate(chunks):
                rows.append({
                    'chunk_id': f"{doc_id}_chunk_{i}",
                    'doc_id': doc_id,
                    'topic': topic,
                    'text': chunk
                })
    
    return pd.DataFrame(rows)