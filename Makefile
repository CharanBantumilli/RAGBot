.PHONY: help install run test lint format clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install RAGBot with dev dependencies
	pip install -e ".[dev]"

run: ## Launch the RAGBot application
	python -m ragbot

test: ## Run the test suite
	pytest tests/ -v --tb=short

lint: ## Check code quality with ruff
	ruff check ragbot/ tests/

format: ## Format code with ruff
	ruff format ragbot/ tests/

clean: ## Remove build artifacts and caches
	rm -rf __pycache__ .pytest_cache dist build *.egg-info chroma_db/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
