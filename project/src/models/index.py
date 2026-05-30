import faiss
import numpy as np
from typing import List, Tuple

class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.chunk_ids = []
        self.doc_ids = []
        self.texts = []
    
    def add(self, vectors: np.ndarray, chunk_ids: List[str], doc_ids: List[str], texts: List[str]) -> None:
        vectors = vectors.astype("float32")
        self.index.add(vectors)
        self.chunk_ids = chunk_ids
        self.doc_ids = doc_ids
        self.texts = texts
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[List[str], List[str], List[float]]:
        query_vector = query_vector.astype("float32")
        distances, indices = self.index.search(query_vector, top_k)
        
        raw_distances = distances[0].tolist()
        
        chunk_ids = [self.chunk_ids[idx] for idx in indices[0]]
        doc_ids = [cid.split('_chunk')[0] for cid in chunk_ids]
        
        return chunk_ids, doc_ids, raw_distances