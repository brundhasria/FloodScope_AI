@echo off
REM FloodScope AI - 1-Click Application Launcher
py scripts\run_app.py %*
if errorlevel 1 (
    python scripts\run_app.py %*
)
