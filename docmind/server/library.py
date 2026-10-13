"""Separate, immutable per-PDF indexes. Never writes the frozen evaluation corpus."""
import copy
import hashlib
import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path

MAX_BYTES = 20 * 1024 * 1024
MAX_PAGES = 200
MAX_CHUNKS = 1200
ID_PATTERN = re.compile(r'^upload_[a-f0-9]{64}$')

class ImportErrorDetail(ValueError):
    pass

class LibraryIntegrityError(ValueError):
    pass

def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    try:
        with temp.open('w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

class Library:
    def __init__(self, directory, encoder=None, exclusive=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._owner=None
        if exclusive:
            from .task_store import TaskStore
            self._owner=TaskStore(self.directory,'library')
            self._owner.acquire()
        self._lock = threading.RLock()
        self._busy = False
        from .embedding_pool import EmbeddingPool
        self.embedding=EmbeddingPool(encoder)
        self._records = {}
        self.recovery_errors = []
        for p in self.directory.glob('upload_*/metadata.json'):
            try:
                record = json.loads(p.read_text(encoding='utf-8'))
                if not ID_PATTERN.fullmatch(p.parent.name) or record['doc_id'] != p.parent.name or record.get('sha256')!=p.parent.name[7:]:
                    raise ValueError('Invalid document identity')
                if record['status'] not in {'ready','failed','queued','parsing','indexing'}:
                    raise ValueError('Invalid import status')
                if record['status'] == 'ready':
                    if set(record.get('artifacts',{})) != {'source.pdf','chunks.json','bm25.pkl','vectors.npz'}:
                        raise ValueError('Incomplete index fingerprints')
                    for name in ('source.pdf','chunks.json','bm25.pkl','vectors.npz'):
                        if not (p.parent/name).is_file(): raise ValueError('Missing index artifact: '+name)
                    for name,digest in record.get('artifacts',{}).items():
                        if name not in {'source.pdf','chunks.json','bm25.pkl','vectors.npz'} or hashlib.sha256((p.parent/name).read_bytes()).hexdigest()!=digest:
                            raise ValueError('Changed index artifact: '+name)
                elif record['status'] != 'failed':
                    record.update(status='failed', message='导入被重启中断或此前失败，请重新上传。')
                    atomic_json(p, record)
                self._records[record['doc_id']] = record
            except Exception as exc:
                self.recovery_errors.append(f'{p.parent.name}: {exc}')

    def records(self):
        with self._lock:
            return copy.deepcopy(list(self._records.values()))

    def record(self, doc_id):
        with self._lock:
            return copy.deepcopy(self._records.get(doc_id))

    def path(self, doc_id):
        if not isinstance(doc_id, str) or not ID_PATTERN.fullmatch(doc_id):
            return None
        return self.directory/doc_id

    def verified_path(self, doc_id, artifact):
        record=self.record(doc_id)
        if not record or record['status']!='ready':
            raise LibraryIntegrityError('论文不存在或尚未就绪')
        if artifact not in {'source.pdf','chunks.json','bm25.pkl','vectors.npz'}:
            raise LibraryIntegrityError('Unknown artifact')
        path=self.path(doc_id)/artifact
        digest=record.get('artifacts',{}).get(artifact)
        if not digest or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise LibraryIntegrityError('论文原文或索引已改变，请恢复原文件或重新导入：'+record['file_name'])
        return path

    def _update(self, doc_id, **changes):
        with self._lock:
            record = {**self._records[doc_id], **changes}
            atomic_json(self.path(doc_id)/'metadata.json', record)
            self._records[doc_id] = record

    def submit(self, body, file_name):
        if not body or len(body) > MAX_BYTES:
            raise ImportErrorDetail('PDF 大小必须在 1 字节到 20 MB 之间。')
        if not body.startswith(b'%PDF-'):
            raise ImportErrorDetail('文件内容不是 PDF。')
        name = Path(file_name.replace('\\','/')).name
        if not name.lower().endswith('.pdf') or len(name) > 180 or any(ord(c)<32 for c in name):
            raise ImportErrorDetail('请选择名称不超过 180 字符的 PDF 文件。')
        digest = hashlib.sha256(body).hexdigest()
        doc_id = 'upload_'+digest
        with self._lock:
            existing = self._records.get(doc_id)
            if existing and existing['status'] != 'failed':
                return {**copy.deepcopy(existing), 'duplicate':True}, 200
            if self._busy:
                return {'error':'import_busy','message':'已有论文正在导入，请完成后再上传。'},429
            if self.recovery_errors:
                return {'error':'library_unavailable','message':'资料库恢复失败，请检查日志。'},503
            directory = self.path(doc_id)
            directory.mkdir(exist_ok=True)
            # Hash identity, never user-supplied filename, controls the storage path.
            (directory/'source.pdf').write_bytes(body)
            record = {'doc_id':doc_id,'sha256':digest,'file_name':name,'status':'queued',
                      'chunks':0,'pages':None,'origin':'uploaded','message':'等待解析',
                      'created_at':datetime.now().isoformat(timespec='seconds'),
                      'retrieval':'BM25 + bge-m3 + RRF'}
            atomic_json(directory/'metadata.json',record)
            self._records[doc_id] = record
            self._busy = True
            threading.Thread(target=self._import,args=(doc_id,),daemon=True,name='docmind-import').start()
            return copy.deepcopy(record),202

    def _import(self, doc_id):
        directory = self.path(doc_id)
        try:
            import pymupdf
            self._update(doc_id,status='parsing',message='解析 PDF 文本与页码')
            with pymupdf.open(directory/'source.pdf') as pdf:
                if pdf.needs_pass: raise ImportErrorDetail('PDF 已加密，请先解密后上传。')
                if not 0 < len(pdf) <= MAX_PAGES: raise ImportErrorDetail('每份 PDF 最多支持 200 页。')
                pages = len(pdf)
                # Explicitly reject image-only documents; no hidden OCR claim.
                if sum(len(page.get_text().strip()) for page in pdf) < 80:
                    raise ImportErrorDetail('未提取到足够文本，可能是扫描件；当前不支持 OCR。')
            from docmind.ingest.pdf_parser import parse_pdf
            from docmind.ingest.chunker import chunk_by_title
            parsed = parse_pdf(directory/'source.pdf')
            name = self.record(doc_id)['file_name']
            for i,b in enumerate(parsed.blocks):
                b.doc_id=doc_id; b.file_name=name; b.block_id=f'{doc_id}-p{b.page}-{i}'
            chunks = chunk_by_title(parsed.blocks)
            if not chunks: raise ImportErrorDetail('解析后没有可检索文本片段。')
            if len(chunks)>MAX_CHUNKS: raise ImportErrorDetail('文本片段过多，当前最多支持 1200 个片段。')
            atomic_json(directory/'chunks.json',[c.to_dict() for c in chunks])
            self._update(doc_id,status='indexing',pages=pages,chunks=len(chunks),message='建立关键词与语义索引（CPU），请稍候')
            from docmind.retrieve.bm25 import BM25Retriever
            from docmind.retrieve.vector import VectorRetriever
            BM25Retriever(chunks).save(directory/'bm25.pkl')
            import numpy as np
            batches=[]
            for start in range(0,len(chunks),8):
                batches.append(self.embedding.encode([c.text for c in chunks[start:start+8]],show_progress=False))
                self._update(doc_id,processed_chunks=min(start+8,len(chunks)),message=f'语义索引 {min(start+8,len(chunks))}/{len(chunks)} 个片段')
            vectors=np.concatenate(batches)
            if vectors.shape != (len(chunks),1024) or not np.isfinite(vectors).all():
                raise ImportErrorDetail('语义索引维度或数值异常，未发布索引。')
            VectorRetriever(vectors,[c.chunk_id for c in chunks]).save(directory/'vectors.npz')
            artifacts={name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
                       for name in ('source.pdf','chunks.json','bm25.pkl','vectors.npz')}
            self._update(doc_id,status='ready',artifacts=artifacts,message='已就绪，可以选择此论文提问')
        except Exception as exc:
            try: self._update(doc_id,status='failed',message=f'导入失败：{type(exc).__name__}: {exc}')
            except Exception as store_exc:
                with self._lock:
                    self.recovery_errors.append(str(store_exc))
                    self._records[doc_id].update(status='failed',message='导入状态保存失败，请检查本机磁盘。')
        finally:
            with self._lock: self._busy=False

    def chunks(self, doc_id):
        record=self.record(doc_id)
        if not record or record['status']!='ready': return []
        self.verified_path(doc_id,'source.pdf')
        return json.loads(self.verified_path(doc_id,'chunks.json').read_text(encoding='utf-8'))
