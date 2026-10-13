"""Deterministic placeholders ONLY for explicit stub import demonstrations."""
import hashlib
import numpy as np


class DemoEncoder:
    def encode(self, texts, **kwargs):
        vectors = np.zeros((len(texts), 1024), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i, int.from_bytes(hashlib.sha256(text.encode()).digest()[:2], 'big') % 1024] = 1
        return vectors
