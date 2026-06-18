# === Makefile ===
# Common commands for the RAG Chatbot project
# Usage: make <target>
# Run 'make help' to see all targets

.PHONY: help setup test lint format typecheck run ingest docker-build docker-run clean

# Default target
help:
	@echo "RAG Chatbot - Available Commands:"
	@echo ""
	@echo "  setup         Create venv, install dependencies (Phase 1)"
	@echo "  test          Run all tests with coverage"
	@echo "  lint          Run ruff linter"
	@echo "  format        Format code with ruff"
	@echo "  typecheck     Run mypy type checker"
	@echo "  run           Start Streamlit app locally"
	@echo "  ingest        Process PDFs -> ChromaDB (Phase 2-3)"
	@echo "  docker-build  Build Docker image"
	@echo "  docker-run    Run with docker-compose (app + Ollama)"
	@echo "  clean         Remove cache, build artifacts"
	@echo ""

# === Local Development ===
setup:
	@echo "[SETUP] Setting up development environment..."
	@python -m venv .venv || uv venv
	@. .venv/bin/activate && pip install --upgrade pip && pip install -e ".[dev]"
	@echo "Setup complete. Activate with: source .venv/bin/activate"

test:
	@echo "[TEST] Running tests..."
	@pytest --cov=src --cov-report=term-missing

lint:
	@echo "[LINT] Linting with ruff..."
	@ruff check src tests

format:
	@echo "[FORMAT] Formatting with ruff..."
	@ruff format src tests

typecheck:
	@echo "[TYPECHECK] Type checking with mypy..."
	@mypy src

run:
	@echo "[START] Starting Streamlit app..."
	@streamlit run src/ui/streamlit_app.py

ingest:
	@echo "[INGEST] Ingesting PDFs into ChromaDB..."
	@python scripts/ingest.py

# === Docker ===
docker-build:
	@echo "[DOCKER] Building Docker image..."
	@docker build -t rag-chatbot:latest .

docker-run:
	@echo "[DOCKER] Starting docker-compose (app + Ollama)..."
	@docker-compose up -d
	@echo "App: http://localhost:8501 | Ollama: http://localhost:11434"

docker-logs:
	@docker-compose logs -f

docker-stop:
	@docker-compose down

# === Maintenance ===
clean:
	@echo "[CLEAN] Cleaning up..."
	@rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete