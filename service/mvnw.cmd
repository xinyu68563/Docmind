@echo off
setlocal EnableDelayedExpansion
set "MVN=%~dp0..\.tools\apache-maven-3.9.11\bin\mvn.cmd"
if exist "%MVN%" (
  call "%MVN%" -Dmaven.repo.local="%~dp0..\.tools\m2" %*
  exit /b !errorlevel!
)
where mvn.cmd >nul 2>nul
if errorlevel 1 (
  echo Apache Maven missing. Install it and add bin to PATH. See docs/TRYOUT.md.
  exit /b 127
)
call mvn.cmd %*
exit /b !errorlevel!
