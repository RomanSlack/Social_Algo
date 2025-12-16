.PHONY: setup gen_data derive train_baseline train_gnn serve demo test clean help

# Default target
help:
	@echo "Calendar-Aware Social Event Feed Recommender"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup           - Install dependencies (run first)"
	@echo "  make gen_data        - Generate synthetic data"
	@echo "  make derive          - Derive availability features"
	@echo "  make build_candidates - Build candidate features"
	@echo "  make train_baseline  - Train two-tower + reranker"
	@echo "  make train_gnn       - Train GNN model"
	@echo "  make serve           - Start web server"
	@echo "  make demo            - Run full pipeline (small mode)"
	@echo "  make demo_full       - Run full pipeline (full mode)"
	@echo "  make test            - Run tests"
	@echo "  make clean           - Remove generated files"
	@echo ""
	@echo "Docker commands:"
	@echo "  make docker_build    - Build Docker image"
	@echo "  make docker_demo     - Run full demo in Docker"
	@echo "  make docker_serve    - Run server in Docker"

# Python executable
PYTHON ?= python

# Small mode flag (set SMALL=1 for quick testing)
ifdef SMALL
SMALL_FLAG = --small
else
SMALL_FLAG =
endif

# Setup virtual environment and install dependencies
setup:
	@echo "Setting up environment..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "Installing PyTorch with CUDA..."
	$(PYTHON) -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	@echo "Installing PyTorch Geometric..."
	$(PYTHON) -m pip install torch_geometric
	$(PYTHON) -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
	@echo "Setup complete!"

# Generate synthetic data
gen_data:
	@echo "Generating synthetic data..."
	$(PYTHON) -m src.data.gen_synth $(SMALL_FLAG)
	@echo "Data generation complete!"

# Derive availability features
derive:
	@echo "Deriving availability features..."
	$(PYTHON) -m src.data.derive_availability $(SMALL_FLAG)
	@echo "Availability derivation complete!"

# Build candidate features
build_candidates:
	@echo "Building candidate features..."
	$(PYTHON) -m src.data.build_candidates $(SMALL_FLAG)
	@echo "Candidate features built!"

# Train two-tower baseline model
train_baseline:
	@echo "Training two-tower model..."
	$(PYTHON) -m src.train.train_two_tower $(SMALL_FLAG)
	@echo "Two-tower training complete!"

# Train GNN model
train_gnn:
	@echo "Training GNN model..."
	$(PYTHON) -m src.train.train_gnn $(SMALL_FLAG)
	@echo "GNN training complete!"

# Start web server
serve:
	@echo "Starting web server at http://localhost:8000"
	$(PYTHON) -m src.serve.app $(SMALL_FLAG)

# Run full demo pipeline (small mode for speed)
demo:
	@echo "Running full demo pipeline (small mode)..."
	@echo "Step 1/6: Generating data..."
	$(PYTHON) -m src.data.gen_synth --small
	@echo "Step 2/6: Deriving availability..."
	$(PYTHON) -m src.data.derive_availability --small
	@echo "Step 3/6: Building candidates..."
	$(PYTHON) -m src.data.build_candidates --small
	@echo "Step 4/6: Training two-tower..."
	$(PYTHON) -m src.train.train_two_tower --small
	@echo "Step 5/6: Training GNN..."
	$(PYTHON) -m src.train.train_gnn --small
	@echo "Step 6/6: Starting server..."
	@echo ""
	@echo "Demo ready! Open http://localhost:8000 in your browser"
	@echo ""
	$(PYTHON) -m src.serve.app --small

# Run full pipeline with full data
demo_full:
	@echo "Running full pipeline (full mode - this will take longer)..."
	$(MAKE) gen_data
	$(MAKE) derive
	$(MAKE) build_candidates
	$(MAKE) train_baseline
	$(MAKE) train_gnn
	$(MAKE) serve

# Run tests
test:
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/ -v --tb=short
	@echo "Tests complete!"

# Smoke test
smoke_test:
	@echo "Running smoke tests..."
	$(PYTHON) -m pytest tests/test_smoke.py -v
	@echo "Smoke tests complete!"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -rf data/*.parquet
	rm -rf models/*.pt
	rm -rf runs/*.json
	rm -rf __pycache__ src/__pycache__ src/*/__pycache__
	rm -rf .pytest_cache
	@echo "Clean complete!"

# Docker commands
docker_build:
	@echo "Building Docker image..."
	docker build -t social-event-recommender -f docker/Dockerfile .
	@echo "Docker build complete!"

docker_demo:
	@echo "Running demo in Docker..."
	cd docker && docker-compose --profile demo up --build demo

docker_serve:
	@echo "Starting server in Docker..."
	cd docker && docker-compose up --build app

docker_pipeline:
	@echo "Running full pipeline in Docker..."
	cd docker && docker-compose --profile pipeline up --build pipeline

# Format code
format:
	@echo "Formatting code..."
	$(PYTHON) -m black src/ tests/
	$(PYTHON) -m isort src/ tests/
	@echo "Formatting complete!"

# Type check
typecheck:
	@echo "Running type checker..."
	$(PYTHON) -m mypy src/ --ignore-missing-imports
	@echo "Type check complete!"
