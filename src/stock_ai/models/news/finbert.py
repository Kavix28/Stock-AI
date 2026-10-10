"""
FinBERT News Encoder for Stock-AI.

Extracts dense 768-dimensional contextual representations from financial headlines
using the pretrained ProsusAI/finbert transformer model.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class FinBertNewsEncoder:
    """
    Pretrained FinBERT feature extractor for financial news headlines.
    
    Extracts the 768-dimensional contextual embedding ([CLS] token or mean-pooled)
    from the transformer backbone.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: Optional[str] = None,
        max_length: int = 128,
        pooling_method: str = "cls",
        lazy_load: bool = True
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.pooling_method = pooling_method.lower()
        self.embedding_dim = 768
        
        if self.pooling_method not in ["cls", "mean"]:
            raise ValueError(f"Unsupported pooling method: {pooling_method}. Choose 'cls' or 'mean'.")

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = None
        self.model = None
        self._is_loaded = False

        if not lazy_load:
            self.load_model()

    def is_loaded(self) -> bool:
        return self._is_loaded

    def load_model(self) -> None:
        """Loads tokenizer and weights from Hugging Face cache or downloads them."""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "transformers is not installed in the current environment. "
                "Install it via `pip install transformers`."
            )

        if self._is_loaded:
            return

        logger.info(f"Loading FinBERT model '{self.model_name}' on device '{self.device}'...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            logger.info("FinBERT model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            raise RuntimeError(f"Could not load FinBERT model '{self.model_name}': {e}") from e

    def get_zero_news_vector(self) -> np.ndarray:
        """Returns deterministic zero-vector representing sessions with no permitted news."""
        return np.zeros(self.embedding_dim, dtype=np.float32)

    def encode_text(self, text: Optional[str]) -> np.ndarray:
        """Encodes a single news headline or concatenated string of headlines."""
        if not text or not text.strip():
            return self.get_zero_news_vector()

        embeddings = self.encode_batch([text], batch_size=1)
        return embeddings[0]

    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encodes a list of headlines into a (N, 768) float32 numpy matrix.
        Empty or whitespace-only texts are mapped to the zero-news vector.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        if not self._is_loaded:
            self.load_model()

        results = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        
        # Identify non-empty texts to batch
        valid_indices = []
        valid_texts = []
        for idx, t in enumerate(texts):
            if t and isinstance(t, str) and t.strip():
                valid_indices.append(idx)
                valid_texts.append(t.strip())

        if not valid_indices:
            return results

        # Process valid texts in mini-batches
        for start_idx in range(0, len(valid_texts), batch_size):
            end_idx = min(start_idx + batch_size, len(valid_texts))
            batch_slice = valid_texts[start_idx:end_idx]
            batch_target_indices = valid_indices[start_idx:end_idx]

            encoded_inputs = self.tokenizer(
                batch_slice,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded_inputs)

                if self.pooling_method == "cls":
                    # [CLS] token is the first token embedding (index 0)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :]
                elif self.pooling_method == "mean":
                    # Attention-masked mean pooling
                    attention_mask = encoded_inputs["attention_mask"].unsqueeze(-1)
                    sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask, dim=1)
                    sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                    batch_embeddings = sum_embeddings / sum_mask

                batch_np = batch_embeddings.cpu().to(torch.float32).numpy()

            for i, target_idx in enumerate(batch_target_indices):
                results[target_idx] = batch_np[i]

        return results

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "max_length": self.max_length,
            "pooling_method": self.pooling_method,
            "device": str(self.device),
            "is_loaded": self._is_loaded
        }
