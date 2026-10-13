"""Read-only preflight checks; --real also checks model/GPU availability."""
import argparse, importlib.util, shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--real',action='store_true');args=p.parse_args()
root=Path(__file__).resolve().parents[1]
missing=[]
for cmd in ['java','node','npm']:
    ok=bool(shutil.which(cmd) or shutil.which(cmd+'.cmd'));print(cmd, 'found' if ok else 'MISSING')
    if not ok:missing.append(cmd)
for module in ['pymupdf','numpy','rank_bm25','jieba']:
    ok=importlib.util.find_spec(module) is not None;print(module, 'found' if ok else 'MISSING')
    if not ok:missing.append(module)
if not (shutil.which('mvn') or shutil.which('mvn.cmd') or (root/'.tools/apache-maven-3.9.11/bin/mvn.cmd').exists()):
    missing.append('Maven');print('Maven MISSING: install Apache Maven and add its bin to PATH')
if args.real:
    import torch
    print('torch',torch.__version__,'CUDA available',torch.cuda.is_available())
    if not torch.cuda.is_available():missing.append('CUDA GPU')
    for name in ['bge-m3','Qwen2.5-7B-Instruct']:
        if not (root/'data/models'/name/'config.json').exists():missing.append('model '+name)
raise SystemExit(1 if missing else 0)
