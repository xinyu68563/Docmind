"""Build a whitelist-only source archive; never packages runtime/user data."""
import hashlib, json, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TESTS=('test_first_use.py','test_chunker.py','test_agent.py','test_round13_worker_boundary.py','test_task_persistence.py','test_embedding_pool.py')
SCRIPTS=('start_docmind.ps1','build_frontend.py','download_models.py','download_llm.py','make_demo_pdf.py','check_environment.py','package_release.py')

def selected_files():
    files=[]
    for directory in ('docmind','frontend/src','service/src/main/java','service/src/test/java','docs','.github'):
        for p in (ROOT/directory).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                files.append(p)
    for name in ('README.md','LICENSE','THIRD_PARTY_NOTICES.md','.gitignore','requirements.txt','requirements-demo.txt','requirements-dev.txt','frontend/index.html','frontend/vite.config.js','frontend/package.json','frontend/package-lock.json','service/pom.xml','service/mvnw','service/mvnw.cmd','service/src/main/resources/application.yml'):
        files.append(ROOT/name)
    files += [ROOT/'scripts'/n for n in SCRIPTS]
    files += [ROOT/'tests'/n for n in TESTS]
    return sorted(set(files))

def main():
    out=ROOT/'dist/releases';out.mkdir(parents=True,exist_ok=True)
    archive=out/'docmind-0.1.0-local-source.zip'
    manifest=[]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in selected_files():
            name=p.relative_to(ROOT).as_posix()
            if p.is_symlink() or not p.is_file():raise ValueError('Invalid release file: '+name)
            data=p.read_bytes();z.writestr('Docmind/'+name,data)
            manifest.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    (out/'source-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    (out/'SHA256SUMS').write_text(digest+'  '+archive.name+'\n',encoding='utf-8')
    print(json.dumps({'archive':str(archive),'files':len(manifest),'bytes':archive.stat().st_size,'sha256':digest}))
if __name__=='__main__':main()
