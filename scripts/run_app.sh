#!/usr/bin/env bash
# Local demo: bind to localhost only (the committed .streamlit/config.toml stays cloud-safe).  Usage: scripts/run_app.sh [port]
cd "$(dirname "$0")/.." && exec streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port "${1:-8501}"
