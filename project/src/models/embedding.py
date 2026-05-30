from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False
        ).astype("float32")