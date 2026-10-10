import sys
import os
import numpy as np

# Add src to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.stock_ai.models.news.finbert import FinBertNewsEncoder

print("Starting FinBERT smoke test...")
encoder = FinBertNewsEncoder(lazy_load=False)
headline = "Apple reports record quarterly revenue and surges in after-hours trading."
emb = encoder.encode_text(headline)
norm = float(np.linalg.norm(emb))

print(f"Successfully encoded headline: '{headline}'")
print(f"Embedding shape: {emb.shape}")
print(f"Embedding dtype: {emb.dtype}")
print(f"L2 Norm: {norm:.4f}")

zero_vec = encoder.encode_text("")
print(f"Zero vector norm: {float(np.linalg.norm(zero_vec))}")
print("SMOKE TEST COMPLETED SUCCESSFULLY.")
