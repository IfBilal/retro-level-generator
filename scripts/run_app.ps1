# Local demo: bind to localhost only.  Usage: pwsh scripts/run_app.ps1 [-Port 8501]
param([int]$Port = 8501)
Set-Location (Join-Path $PSScriptRoot "..")
streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port $Port
