# Needs Python 3.10+. Override if your default is older: make install PYTHON=python3.12
PYTHON ?= python3
PY := .venv/bin/python

.PHONY: install api ui mcp test lint eval samples

install:
	$(PYTHON) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	@test -f .env || cp .env.example .env

api:
	$(PY) -m uvicorn app.main:app --port 8000

ui:
	$(PY) -m streamlit run ui/streamlit_app.py

mcp:
	$(PY) -m app.mcp_server

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

eval:
	$(PY) -m eval.run_eval

samples:
	$(PY) scripts/make_sample_docs.py
