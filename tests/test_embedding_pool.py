import threading
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from docmind.server.embedding_pool import EmbeddingPool

def test_shared_lazy_initialization_and_call_serialization():
    lock=threading.Lock();counts={'created':0,'active':0,'max_active':0}
    class Model:
        def encode(self,texts,**kwargs):
            with lock:
                counts['active']+=1;counts['max_active']=max(counts['max_active'],counts['active'])
            time.sleep(.01)
            with lock:counts['active']-=1
            return np.zeros((len(texts),1024))
        def encode_query(self,query):return self.encode([query])[0]
    def factory():counts['created']+=1;time.sleep(.01);return Model()
    pool=EmbeddingPool(factory=factory)
    assert not pool.describe()['loaded']
    with ThreadPoolExecutor(max_workers=6) as executor:
        outputs=list(executor.map(lambda i:pool.encode_query('q') if i%2 else pool.encode(['doc']),range(12)))
    assert len(outputs)==12 and counts['created']==1 and counts['max_active']==1
    assert pool.describe()['loaded']
    with pytest.raises(ValueError):pool.to('cuda')

def test_failed_model_initialization_can_retry():
    calls=[]
    class Model:
        def encode(self,*args,**kwargs):return 'ok'
    def factory():
        calls.append(1)
        if len(calls)==1:raise RuntimeError('load failed')
        return Model()
    pool=EmbeddingPool(factory=factory)
    with pytest.raises(RuntimeError):pool.encode(['doc'])
    assert not pool.describe()['loaded']
    assert pool.encode(['doc'])=='ok'
