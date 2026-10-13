"""Read-only corpus catalogue; document/chunk identifiers never become arbitrary paths."""
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
library = None

@lru_cache(maxsize=1)
def corpus():
    path = ROOT/'data/processed/chunks.json'
    if not path.exists():
        if any((ROOT/'data/processed/index'/name).exists() for name in ('bm25.pkl', 'vectors.npz')):
            raise ValueError('内置索引不完整：缺少 chunks.json')
        return []
    return json.loads(path.read_text(encoding='utf-8'))

def documents():
    docs = {}
    for c in corpus():
        d = docs.setdefault(c['doc_id'], {'doc_id':c['doc_id'], 'file_name':c['file_name'], 'chunks':0})
        d['chunks'] += 1
    return [{**d, 'status':'ready', 'origin':'builtin'} for d in docs.values()] + (library.records() if library else [])

def document_path(doc_id):
    if library and library.record(doc_id):
        return library.verified_path(doc_id,'source.pdf')
    doc = next((d for d in documents() if d['doc_id']==doc_id), None)
    if doc is None:
        return None
    raw = (ROOT/'data/raw').resolve()
    p = (raw/doc['file_name']).resolve()
    return p if p.parent==raw and p.suffix.lower()=='.pdf' and p.is_file() else None

def chunk(chunk_id):
    found = next((c for c in corpus() if c['chunk_id']==chunk_id), None)
    if found or not library: return found
    for d in library.records():
        if chunk_id.startswith(d['doc_id']+'-'):
            return next((c for c in library.chunks(d['doc_id']) if c['chunk_id']==chunk_id),None)
    return None


def render_page(doc_id, page):
    path=document_path(doc_id)
    if path is None:raise ValueError('Document not found')
    return _page_image(str(path),path.stat().st_mtime_ns,page)

@lru_cache(maxsize=8)
def _page_image(path,mtime,page):
    import pymupdf
    if not isinstance(page,int) or not 1<=page<=200:raise ValueError('页码必须在 1 到 200 之间')
    with pymupdf.open(path) as pdf:
        if pdf.needs_pass:raise ValueError('PDF 已加密')
        if page>len(pdf):raise ValueError('页码超出文档范围')
        rect=pdf[page-1].rect
        if rect.width<=0 or rect.height<=0:raise ValueError('PDF 页面尺寸异常')
        scale=min(1400/rect.width,2200/rect.height,2)
        pix=pdf[page-1].get_pixmap(matrix=pymupdf.Matrix(scale,scale),alpha=False)
        return pix.tobytes('png'),len(pdf)
