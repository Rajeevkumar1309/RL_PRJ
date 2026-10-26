@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found.
    echo Create it with: py -m venv .venv
    echo Then install dependencies with: .venv\Scripts\python.exe -m pip install -r requirements.txt
    exit /b 1
)

".venv\Scripts\python.exe" cartpole_demo.py --render
