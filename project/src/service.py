import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("faiss.loader").setLevel(logging.ERROR)

import math
import sys
from pathlib import Path
import subprocess
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import importlib

from utils.chunker import prepare_chunks
from models.embedding import EmbeddingModel
from models.index import VectorIndex

from configs.config import DATA_PATH, CHUNK_SIZE, OVERLAP, TOP_K, MODEL_NAME, DEVICE

app = FastAPI(title="Сервис быстрого поиска ответов по запросам пользователей")

embedder = None
index = None
df_chunks = None
startup_error = None

class PredictRequest(BaseModel):
    query: str
    top_k: Optional[int] = TOP_K

class PredictResponse(BaseModel):
    query: str
    predictions: List[dict]

def check_and_install_dependencies() -> bool:
    missing = []
    required = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'faiss': 'faiss-cpu',
        'sentence_transformers': 'sentence-transformers',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'pydantic': 'pydantic'
    }
    
    for module_name, package_name in required.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)
            logger.warning(f"{package_name} отсутствует")
    
    if missing:
        logger.warning(f"Отсутствуют библиотеки: {missing}")
        for package in missing:
            logger.info(f"Установка {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    
    return True

@app.on_event("startup")
async def startup():
    global embedder, index, df_chunks, startup_error
    
    try:
        logger.info("Запуск сервиса")
        
        logger.info("Проверка зависимостей...")
        if not check_and_install_dependencies():
            raise RuntimeError("Не удалось установить зависимости")
        logger.info("Зависимости установлены")
        
        logger.info("Проверка файла документов...")
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Файл не найден: {DATA_PATH}")
        logger.info(f"Файл найден: {DATA_PATH}")
        
        logger.info("Загрузка документов...")
        df_docs = pd.read_csv(DATA_PATH)
        logger.info(f"Загружено документов: {len(df_docs)}")
        
        logger.info("Подготовка чанков...")
        df_chunks = prepare_chunks(df_docs, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        logger.info(f"Количество чанков: {len(df_chunks)}")
        
        logger.info("Загрузка модели эмбеддингов...")
        embedder = EmbeddingModel(model_name=MODEL_NAME, device=DEVICE)
        logger.info(f"Модель загружена: {MODEL_NAME}")
        
        logger.info("Векторизация документов...")
        vectors = embedder.encode(df_chunks['text'].tolist())
        logger.info(f"Векторизация завершена, форма: {vectors.shape}")
        
        logger.info("Построение FAISS индекса...")
        index = VectorIndex(dim=vectors.shape[1])
        index.add(
            vectors=vectors,
            chunk_ids=df_chunks['chunk_id'].tolist(),
            doc_ids=df_chunks['doc_id'].tolist(),
            texts=df_chunks['text'].tolist()
        )
        logger.info("Индекс построен")
        
        logger.info("Сервис запущен и готов к работе")

        
    except Exception as e:
        startup_error = str(e)
        logger.error(f"Ошибка запуска: {e}")
        raise e

@app.get("/health")
async def health():
    def check_dependencies() -> List[str]:
        missing = []
        required = {
            'pandas': 'pandas',
            'numpy': 'numpy',
            'faiss': 'faiss-cpu',
            'sentence_transformers': 'sentence-transformers',
            'fastapi': 'fastapi',
            'uvicorn': 'uvicorn',
            'pydantic': 'pydantic'
        }
        for module_name, package_name in required.items():
            try:
                importlib.import_module(module_name)
            except ImportError:
                missing.append(package_name)
    
        return missing
    
    status = {
        "status": "ok",
        "model": MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "documents_loaded": False,
        "index_built": False,
        "dependencies": {}
    }
    
    missing = check_dependencies()
    status["dependencies"]["all_installed"] = len(missing) == 0
    status["dependencies"]["missing"] = missing
    
    status["documents_loaded"] = df_chunks is not None and len(df_chunks) > 0
    status["index_built"] = index is not None
    
    if embedder is not None:
        status["model_loaded"] = True
        status["model_name"] = MODEL_NAME
    else:
        status["model_loaded"] = False
    
    if startup_error:
        status["status"] = "degraded"
        status["startup_error"] = startup_error
    
    status["data_file_exists"] = DATA_PATH.exists()
    status["data_file_path"] = str(DATA_PATH)
    
    return status

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    global embedder, index, df_chunks
    
    logger.info(f"Получен запрос: {request.query}")
    
    if embedder is None or index is None:
        logger.error("Сервис не инициализирован")
        raise HTTPException(
            status_code=503, 
            detail="Сервис не инициализирован. Проверьте /health"
        )
    
    try:
        query_vec = embedder.encode([request.query])
        chunk_ids, doc_ids, distances = index.search(query_vec, top_k=request.top_k)
        
        logger.info(f"Найдено {len(doc_ids)} документов, расстояния: {distances}")
        
        min_dist = min(distances)
        max_dist = max(distances)
        if max_dist > min_dist:
            scores = [1 - (d - min_dist) / (max_dist - min_dist) for d in distances]
        else:
            scores = [0.5] * len(distances)
        
        predictions = []
        for i, (doc_id, score) in enumerate(zip(doc_ids, scores)):
            doc_text = df_chunks[df_chunks['chunk_id'] == chunk_ids[i]]['text'].values
            text = doc_text[0] if len(doc_text) > 0 else ""
            
            predictions.append({
                "rank": i + 1,
                "doc_id": doc_id,
                "text": text,
                "score": round(distances[i], 4)
            })
        
        logger.info(f"Ответ отправлен, документов запрошено: {len(predictions)}")
        return PredictResponse(query=request.query, predictions=predictions)
    
    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()