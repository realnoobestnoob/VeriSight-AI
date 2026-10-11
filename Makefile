.PHONY: install test lint format pipeline demo app mlflow docker
install:
	python -m pip install -e ".[dev]"
test:
	pytest
lint:
	ruff check . && ruff format --check .
format:
	ruff format . && ruff check --fix .
pipeline:
	python -m verisight_ai.cli pipeline --config configs/default.yaml
demo:
	python -m verisight_ai.cli pipeline --config configs/default.yaml --demo
app:
	streamlit run app.py
mlflow:
	mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
docker:
	docker build -t verisight-ai .
