from sentence_transformers import SentenceTransformer
from ..Agent.logger import LOGGER as logger


class EmbeddingManager:
    _instance = None
    _embedding_model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if self._embedding_model is None:
            try:
                EmbeddingManager._embedding_model = SentenceTransformer(
                    'all-MiniLM-L6-v2')
                logger.info("✓ Embedding model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
                EmbeddingManager._embedding_model = None

    def encode(self, text: str):
        if self._embedding_model is None:
            raise ValueError("Embedding model not initialized")
        return self._embedding_model.encode(text)
