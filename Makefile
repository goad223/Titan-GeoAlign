# =============================================================================
# Titan-GeoAlign — Makefile
# =============================================================================
.DEFAULT_GOAL := help
SHELL         := /bin/bash
.SHELLFLAGS   := -eu -o pipefail -c

# Project metadata
PROJECT       := titan-geoalign
SRC_DIR       := src/titan_geoalign
CLI_DIR       := cli
TESTS_DIR     := tests
DOCS_DIR      := docs
DOCKER_IMAGE  := titan-geoalign
DOCKER_TAG    := latest

# Python / pip
PYTHON        ?= python3
PIP           ?= pip

# Colors
BOLD   := \033[1m
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RESET  := \033[0m

##@ Help
.PHONY: help
help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\n$(BOLD)Usage:$(RESET)\n  make $(YELLOW)<target>$(RESET)\n"} \
	      /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2 } \
	      /^##@/ { printf "\n$(BOLD)%s$(RESET)\n", substr($$0, 5) }' $(MAKEFILE_LIST)

##@ Setup
.PHONY: install
install: ## Install runtime dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e . --no-deps

.PHONY: install-dev
install-dev: ## Install all dependencies including dev/test tools
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e ".[dev]" --no-deps
	pre-commit install

.PHONY: install-gpu
install-gpu: ## Install GPU (CUDA 12.4) dependencies on top of the base install
	$(PIP) install -r requirements-gpu.txt

.PHONY: install-conda
install-conda: ## Create / update conda environment from environment.yml
	conda env update --file environment.yml --prune
	@echo "Activate with: conda activate titan-geoalign"

##@ Development
.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

.PHONY: format
format: ## Auto-format code with black and ruff
	black $(SRC_DIR) $(CLI_DIR) $(TESTS_DIR)
	ruff check --fix $(SRC_DIR) $(CLI_DIR) $(TESTS_DIR)

.PHONY: lint
lint: ## Run linters (ruff, black --check, mypy)
	black --check $(SRC_DIR) $(CLI_DIR) $(TESTS_DIR)
	ruff check $(SRC_DIR) $(CLI_DIR) $(TESTS_DIR)
	mypy $(SRC_DIR) $(CLI_DIR)

.PHONY: typecheck
typecheck: ## Run mypy type checker only
	mypy $(SRC_DIR) $(CLI_DIR)

##@ Testing
.PHONY: test
test: ## Run unit tests
	pytest $(TESTS_DIR) -m "not integration and not gpu and not slow" -v

.PHONY: test-all
test-all: ## Run all tests
	pytest $(TESTS_DIR) -v

.PHONY: test-integration
test-integration: ## Run integration tests
	pytest $(TESTS_DIR) -m integration -v

.PHONY: test-gpu
test-gpu: ## Run GPU-specific tests (requires CUDA device)
	pytest $(TESTS_DIR) -m gpu -v

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	pytest $(TESTS_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing -v
	@echo "Coverage report: htmlcov/index.html"

.PHONY: test-parallel
test-parallel: ## Run tests in parallel (requires pytest-xdist)
	pytest $(TESTS_DIR) -n auto -v

##@ Build
.PHONY: build
build: clean ## Build Python distribution packages (wheel + sdist)
	$(PYTHON) -m hatch build

.PHONY: build-wheel
build-wheel: ## Build wheel only
	$(PYTHON) -m hatch build --target wheel

##@ Docker
.PHONY: docker-build
docker-build: ## Build the Docker image
	docker build \
	  --build-arg BUILD_DATE="$(shell date -u +'%Y-%m-%dT%H:%M:%SZ')" \
	  --build-arg VCS_REF="$(shell git rev-parse --short HEAD)" \
	  -t $(DOCKER_IMAGE):$(DOCKER_TAG) \
	  -f docker/Dockerfile .

.PHONY: docker-build-gpu
docker-build-gpu: ## Build the GPU-enabled Docker image
	docker build \
	  --build-arg BUILD_DATE="$(shell date -u +'%Y-%m-%dT%H:%M:%SZ')" \
	  --build-arg VCS_REF="$(shell git rev-parse --short HEAD)" \
	  -t $(DOCKER_IMAGE):$(DOCKER_TAG)-gpu \
	  -f docker/Dockerfile.gpu .

.PHONY: docker-run
docker-run: ## Run the Docker container interactively
	docker run --rm -it \
	  -v "$(PWD)/data:/app/data" \
	  -p 8000:8000 \
	  $(DOCKER_IMAGE):$(DOCKER_TAG)

.PHONY: docker-push
docker-push: ## Push Docker image to registry (set REGISTRY env var)
	docker tag $(DOCKER_IMAGE):$(DOCKER_TAG) $(REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)
	docker push $(REGISTRY)/$(DOCKER_IMAGE):$(DOCKER_TAG)

##@ Documentation
.PHONY: docs-serve
docs-serve: ## Serve documentation locally with live reload
	mkdocs serve --dev-addr 0.0.0.0:8001

.PHONY: docs-build
docs-build: ## Build static documentation site
	mkdocs build --strict

.PHONY: docs-deploy
docs-deploy: ## Deploy documentation to GitHub Pages
	mkdocs gh-deploy --force

##@ Utilities
.PHONY: data-dirs
data-dirs: ## Create required data and model directories
	mkdir -p data models outputs logs
	touch data/.gitkeep models/.gitkeep outputs/.gitkeep logs/.gitkeep

.PHONY: clean
clean: ## Remove build artifacts, caches, and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type f -name "coverage.xml" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov
	rm -rf dist build
	rm -rf *.egg-info src/*.egg-info
	rm -rf site  # mkdocs output

.PHONY: clean-all
clean-all: clean ## Remove everything including conda env and node_modules
	rm -rf .venv node_modules
	conda env remove -n titan-geoalign 2>/dev/null || true

.PHONY: version
version: ## Show current project version
	@$(PYTHON) -c "import importlib.metadata; print(importlib.metadata.version('titan-geoalign'))"

.PHONY: env-info
env-info: ## Display Python and key dependency versions
	@$(PYTHON) --version
	@$(PIP) --version
	@$(PYTHON) -c "import torch; print('torch:', torch.__version__)"
	@$(PYTHON) -c "import numpy; print('numpy:', numpy.__version__)"
	@$(PYTHON) -c "import rasterio; print('rasterio:', rasterio.__version__)"
	@$(PYTHON) -c "import gdal; from osgeo import gdal; print('gdal:', gdal.__version__)" 2>/dev/null || true
