"""Download model snapshots from their official Hugging Face repositories."""
from pathlib import Path
from huggingface_hub import snapshot_download
root=Path(__file__).resolve().parents[1]/'data/models'
for repo,dirname in [('BAAI/bge-m3','bge-m3'),('Qwen/Qwen2.5-7B-Instruct','Qwen2.5-7B-Instruct')]:
    snapshot_download(repo_id=repo, local_dir=root/dirname)
