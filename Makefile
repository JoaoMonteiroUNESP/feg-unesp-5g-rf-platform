.PHONY: install install-dev test test-fast lint run demo clean

PY ?= python -X utf8

install:
	$(PY) -m pip install -r requirements.txt

install-dev:
	$(PY) -m pip install -r requirements-dev.txt

test:
	$(PY) -m pytest -v

test-fast:
	$(PY) -m pytest -v -m "not slow"

lint:
	$(PY) -m ruff check app tests scripts

run:
	$(PY) -m uvicorn app.main:app --reload --port 8000

demo:
	$(PY) scripts/generate_demo_log.py

clean:
	rm -rf data/db data/logs data/raw data/exports
