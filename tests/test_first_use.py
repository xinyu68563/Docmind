import json
import sys
import types
import pytest
from docmind.server import sources
from docmind.server.worker import RealEngine, Worker, ValidationError


def test_empty_catalogue_and_incomplete_index(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, 'ROOT', tmp_path)
    monkeypatch.setattr(sources, 'library', None)
    sources.corpus.cache_clear()
    try:
        assert sources.documents() == []
        (tmp_path/'data/processed/index').mkdir(parents=True)
        (tmp_path/'data/processed/index/vectors.npz').touch()
        sources.corpus.cache_clear()
        with pytest.raises(ValueError, match='不完整'):
            sources.documents()
    finally:
        sources.corpus.cache_clear()


def test_empty_builtin_does_not_prevent_model_bundle(tmp_path, monkeypatch):
    loaded = []
    class LLM:
        def __init__(self, **kw): loaded.append(kw)
    monkeypatch.setitem(sys.modules, 'docmind.generate.llm', types.SimpleNamespace(LocalLLM=LLM))
    engine = RealEngine(chunk_path=tmp_path/'chunks.json', bm25_path=tmp_path/'bm25.pkl', vec_path=tmp_path/'vectors.npz')
    bundle = engine._load()
    assert bundle['chunks'] == [] and bundle['hybrid'] is None
    assert bundle['vector'].vectors.shape == (0, 1024)
    assert len(loaded) == 1
    assert engine._load() is bundle


def test_partial_builtin_index_is_error_before_model(tmp_path):
    (tmp_path/'chunks.json').write_text('[]')
    engine = RealEngine(chunk_path=tmp_path/'chunks.json', bm25_path=tmp_path/'bm25.pkl', vec_path=tmp_path/'vectors.npz')
    with pytest.raises(ValueError, match='不完整'): engine._load()


def test_no_implicit_empty_scope_or_missing_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, 'ROOT', tmp_path)
    monkeypatch.setattr(sources, 'library', None)
    sources.corpus.cache_clear()
    worker = Worker(RealEngine(), journal_dir=tmp_path/'journal')
    try:
        with pytest.raises(ValidationError, match='先上传'):
            worker.submit({'question':'question'})
        with pytest.raises(ValidationError, match='不存在'):
            worker.submit({'question':'question', 'document_ids':['unknown']})
        assert not worker._tasks
    finally:
        sources.corpus.cache_clear()


def test_selected_upload_works_without_builtin(tmp_path, monkeypatch):
    import numpy as np
    from docmind.server.demo_encoder import DemoEncoder
    from docmind.server.library import Library
    import pymupdf, time
    library = Library(tmp_path/'library', DemoEncoder())
    pdf=pymupdf.open();page=pdf.new_page()
    page.insert_text((40,40), 'Original synthetic upload. Spectral sampling convergence estimates. '*12)
    row,_=library.submit(pdf.tobytes(),'synthetic.pdf');pdf.close()
    for _ in range(100):
        row=library.record(row['doc_id'])
        if row['status'] in {'ready','failed'}:break
        time.sleep(.02)
    assert row['status']=='ready', row
    engine=RealEngine();engine.library=library
    hybrid,meta=engine._multi_retrieval({'embedder':library.embedding,'chunks':[]}, [row['doc_id']])
    assert len(hybrid.bm25.blocks)>0
    assert hybrid.vector.vectors.shape[1]==1024
    assert all(v['doc_id']==row['doc_id'] for v in meta.values())

