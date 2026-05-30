from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "documents.csv"

CHUNK_SIZE = 40
OVERLAP = 20

TOP_K = 5

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEVICE = "cpu"