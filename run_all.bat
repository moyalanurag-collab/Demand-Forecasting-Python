@echo off
cd /d "%~dp0"
pip install -r requirements.txt
python -m ipykernel install --user --name python3
python src\build_notebooks.py
echo Done. Open the notebooks in the notebooks folder and the files in reports.
pause
