"""Optional, versioned PDF source reviews; never read questions or evaluation gold.

Original retrieved text stays intact. Independently reviewed source sections are
attached once per document with physical-page and original-PDF fingerprints.
This is targeted corpus enrichment, not automated mathematical OCR.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / 'data/source_reviews/math_v1/notes.json'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source_reviews(path: Path = NOTES, raw_dir: Path | None = None) -> list[dict]:
    if not path.exists():
        return []
    document = json.loads(path.read_text(encoding='utf-8'))
    if document.get('schema') != 'docmind.pdf_source_review/1':
        raise ValueError('Unsupported source review schema')
    raw_dir = raw_dir or ROOT/'data/raw'
    records = document['records']
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate source review identity')
    for record in records:
        name = record['file_name']
        if '/' in name or '\\' in name or not name.endswith('.pdf'):
            raise ValueError('Source review must identify a PDF basename')
        source = raw_dir/name
        if not source.is_file() or sha256(source) != record['pdf_sha256']:
            raise ValueError(f'Source PDF changed or missing: {name}')
        if not isinstance(record['physical_page'],int) or record['physical_page'] < 1:
            raise ValueError('Invalid physical source page')
        if not isinstance(record['text'],str) or not record['text'].strip():
            raise ValueError('Empty source transcription')
        import pymupdf
        with pymupdf.open(source) as document:
            if record['physical_page'] > len(document):
                raise ValueError('Source review page does not exist')
            raw = document[record['physical_page']-1].get_text()
            if hashlib.sha256(raw.encode('utf-8')).hexdigest() != record['extracted_page_sha256']:
                raise ValueError('Source review physical page fingerprint mismatch')
    return records


def reviewed_model_text(raw: str, records: list[dict]) -> str:
    additions = []
    for record in records:
        additions.append(f"[原PDF公式核对补充，物理第{record['physical_page']}页，记录{record['id']}]\n"
                         + record['text'])
    return raw + ('\n\n'+'\n\n'.join(additions) if additions else '')


def prepare_source_evidence(evidence: list[dict], records: list[dict] | None = None) -> list[dict]:
    """Keep original chunk identities and attach reviewed sections once per file."""
    records = load_source_reviews() if records is None else records
    by_file: dict[str,list[dict]] = {}
    for record in records:
        by_file.setdefault(record['file_name'],[]).append(record)
    seen = set()
    result = []
    for original in evidence:
        entry = copy.deepcopy(original)
        # Recompute from original text; never trust previously supplied model_text.
        entry.pop('source_reviews',None)
        entry.pop('model_text',None)
        name = entry.get('file_name') or entry.get('source','').split(' · ',1)[0]
        # Upload names are presentation only; they must not inherit reviews of a
        # different frozen PDF with the same basename.
        if str(entry.get('doc_id','')).startswith('upload_') or str(entry.get('chunk_id','')).startswith('upload_'):
            result.append(entry)
            continue
        if name in by_file and name not in seen:
            reviews = copy.deepcopy(by_file[name])
            entry['source_reviews'] = reviews
            entry['model_text'] = reviewed_model_text(entry['text'],reviews)
            seen.add(name)
        result.append(entry)
    return result


def model_text(entry: dict) -> str:
    return entry.get('model_text',entry['text'])


def formal_source_request(question: str) -> bool:
    import re
    return bool(re.search(r'公式|表示|误差界|估计.*误差|formula|representation|error bound|estimate.*error',question,re.I))


def source_review_options(evidence: list[dict]) -> list[dict]:
    return [{'source_review_id':record['id'], 'evidence_number':i,
             'file_name':record['file_name'], 'physical_page':record['physical_page'], 'text':record['text']}
            for i,entry in enumerate(evidence,1) for record in entry.get('source_reviews',[])]


def render_source_quote(option: dict) -> str:
    number = option['evidence_number']
    return (f"原 PDF 第{option['physical_page']}页核对的完整条件与公式 [{number}]：\n\n"
            + option['text'] + f"\n\n来源：{option['file_name']}，物理第{option['physical_page']}页 [{number}]。")


def format_source_options(options: list[dict]) -> str:
    return '\n\n'.join(f"候选{i}（证据{option['evidence_number']}，{option['file_name']}，物理第{option['physical_page']}页）：\n{option['text']}"
                         for i,option in enumerate(options,1))


def selected_source_option(decision: dict | None, options: list[dict]) -> dict | None:
    choice = decision.get('choice') if isinstance(decision,dict) else None
    return options[choice-1] if type(choice) is int and 1 <= choice <= len(options) else None
