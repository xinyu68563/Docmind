"""Windows launcher regression: rebuilding target must not mutate the live JAR."""
import os
from pathlib import Path
import subprocess
import zipfile
import pytest

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows PowerShell launcher')
ROOT = Path(__file__).resolve().parents[1]


def quote(path):
    return "'" + str(path).replace("'", "''") + "'"


def prepare(jar, logs):
    # Extract the function AST, without executing the script that starts services.
    script = (
        "$ErrorActionPreference='Stop';"
        # Python may inherit PowerShell 7 module paths when spawning Windows PS5.
        "$env:PSModulePath=(Join-Path $PSHOME 'Modules')+[IO.Path]::PathSeparator+$env:PSModulePath;"
        f"$jarPath={quote(jar)};$logDir={quote(logs)};"
        f"$ast=[Management.Automation.Language.Parser]::ParseFile({quote(ROOT/'scripts/start_docmind.ps1')},[ref]$null,[ref]$null);"
        "$fn=$ast.Find({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Prepare-RuntimeJar'},$true);"
        "Invoke-Expression $fn.Extent.Text; Prepare-RuntimeJar"
    )
    return subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', script],
                          capture_output=True, text=True, encoding='utf-8', errors='replace')


def bundle(path):
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('BOOT-INF/lib/dependency.jar', b'test-only structural placeholder')
        archive.writestr('BOOT-INF/classes/static/index.html', '<html>test</html>')


def test_runtime_copy_is_immutable_and_checked(tmp_path):
    jar = tmp_path/'built.jar'
    logs = tmp_path/'logs'
    bundle(jar)
    original = jar.read_bytes()
    first = prepare(jar, logs)
    assert first.returncode == 0, first.stderr
    runtime = Path(first.stdout.strip())
    assert runtime.is_relative_to(logs) and runtime.read_bytes() == original
    assert prepare(jar, logs).stdout.strip() == str(runtime)
    runtime.write_bytes(b'corrupted runtime')
    broken = prepare(jar, logs)
    assert broken.returncode != 0 and 'integrity' in broken.stderr
    runtime.write_bytes(original)
    jar.write_bytes(b'build output replaced')
    assert runtime.read_bytes() == original


def test_plain_jar_rejected_before_starting_java(tmp_path):
    jar = tmp_path/'plain.jar'
    with zipfile.ZipFile(jar, 'w') as archive:
        archive.writestr('application.class', b'test')
    result = prepare(jar, tmp_path/'logs')
    assert result.returncode != 0 and 'incomplete' in result.stderr
    assert not (tmp_path/'logs/runtime').exists()
