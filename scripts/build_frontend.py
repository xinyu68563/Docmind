"""Build the Vue UI and bundle it into the local Spring Boot service."""
from pathlib import Path
import shutil, subprocess

ROOT=Path(__file__).resolve().parents[1]
def main():
    npm=shutil.which('npm.cmd') or shutil.which('npm')
    if not npm:raise SystemExit('Node.js/npm is required')
    subprocess.run([npm,'ci','--no-audit','--no-fund'],cwd=ROOT/'frontend',check=True)
    subprocess.run([npm,'run','build'],cwd=ROOT/'frontend',check=True)
    shutil.copytree(ROOT/'frontend/dist',ROOT/'service/src/main/resources/static',dirs_exist_ok=True)
    print('UI bundled. Rebuild service using service/mvnw.cmd -f service/pom.xml package')
if __name__=='__main__':main()
