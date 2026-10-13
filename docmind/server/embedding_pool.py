"""One CPU embedding model shared by import and inference, with serialized calls."""
import threading

class EmbeddingPool:
    def __init__(self, model=None, factory=None):
        self._model=model
        self._factory=factory
        self._lock=threading.RLock()

    def _load(self):
        if self._model is None:
            if self._factory:
                self._model=self._factory()
            else:
                from docmind.retrieve.embeddings import EmbeddingModel
                self._model=EmbeddingModel(preset='m3',device='cpu',use_fp16=False,batch_size=2)
        return self._model

    def encode(self, texts, **kwargs):
        with self._lock:
            return self._load().encode(texts, **kwargs)

    def encode_query(self, query):
        with self._lock:
            return self._load().encode_query(query)

    def to(self, device):
        if device != 'cpu':raise ValueError('Shared service embedding must remain on CPU')

    def describe(self):
        return {'device':'cpu','loaded':self._model is not None,
                'shared_by':['import','query'],'calls_serialized':True}
