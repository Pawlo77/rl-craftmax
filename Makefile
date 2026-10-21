ifneq ("$(wildcard .env)","")
	include .env
	export
endif

export PYTHONPATH=.
export OMP_NUM_THREADS=1

MLFLOW_HOST ?= 127.0.0.1
MLFLOW_PORT ?= 5005
MLFLOW_WORKERS ?= 1

SKILLS_TESTS := tests/hooks/test_skill_frontmatter.py

PROFILE ?= laptop
QUEUE_MODULE := uv run python -m wut_murl_craftax.orchestration.queue
STORE_MODULE := uv run python -m wut_murl_craftax.orchestration.store
# --method and --expand-fine, when METHOD and FINE are set
PHASE_FLAGS = $(if $(METHOD),--method $(METHOD)) $(if $(FINE),--expand-fine $(FINE))

include kit.mk

.PHONY: coverage merge-mlflow mlflow mlflow-stop queue status train-ppo vis warm

coverage: ## Run tests with coverage report
	uv run pytest --cov=./src/wut_murl_craftax --cov-report=term-missing tests/

warm: ## Build the Craftax texture caches and print the actor/learner devices
	uv run python -m wut_murl_craftax.orchestration.runtime

queue: ## Run a phase: make queue PHASE=tune-coarse [METHOD=ppo] [FINE=ppo] [DRY_RUN=1] [PROFILE=cuda]
	$(if $(PHASE),,$(error PHASE is required, e.g. make queue PHASE=tune-coarse))$(QUEUE_MODULE) --phase configs/phases/$(PHASE).yaml --profile configs/profiles/$(PROFILE).yaml $(PHASE_FLAGS)$(if $(DRY_RUN), --dry-run)

status: ## Pending/running/finished/killed runs of a phase: make status PHASE=tune-coarse [METHOD=ppo] [FINE=ppo]
	$(if $(PHASE),,$(error PHASE is required, e.g. make status PHASE=tune-coarse))$(STORE_MODULE) status --phase configs/phases/$(PHASE).yaml $(PHASE_FLAGS)

merge-mlflow: ## Merge a node's MLflow store into mlruns.db: make merge-mlflow SRC=mlruns-node-3.db
	$(if $(SRC),,$(error SRC is required, e.g. make merge-mlflow SRC=mlruns-node-3.db))$(STORE_MODULE) merge $(SRC)

train-ppo: ## Baseline PPO run: make train-ppo ARGS="--env Craftax-Classic-Symbolic-v1 --steps 1e6"
	uv run python -m wut_murl_craftax.training.ppo $(ARGS)

vis: ## Run one visualisation step: make vis STEP=review ARGS="outputs/visualisation/review/candidates.csv"
	uv run python -m wut_murl_craftax.visualisation.$(STEP) $(ARGS)

mlflow: ## Launch MLflow UI for local runs
	@PORT="$(MLFLOW_PORT)"; \
	if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$$PORT" -sTCP:LISTEN >/dev/null 2>&1; then \
		echo "Port $$PORT is already in use. Run 'make mlflow-stop' or use another port: make mlflow MLFLOW_PORT=5001"; \
		lsof -nP -iTCP:"$$PORT" -sTCP:LISTEN; \
		exit 1; \
	fi; \
	uv run mlflow ui --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns --host "$(MLFLOW_HOST)" --port "$$PORT" --workers "$(MLFLOW_WORKERS)"

mlflow-stop: ## Stop local MLflow UI processes
	@PORT="$(MLFLOW_PORT)"; \
	PIDS="$$( ( \
		pgrep -f 'mlflow.server.fastapi_app' || true; \
		pgrep -f 'python -m mlflow' || true; \
		pgrep -f 'mlflow ui' || true; \
		pgrep -f 'mlflow server' || true \
	) | sort -u )"; \
	if [ -z "$$PIDS" ] && command -v lsof >/dev/null 2>&1; then \
		PORT_PIDS="$$(lsof -tiTCP:"$$PORT" -sTCP:LISTEN 2>/dev/null || true)"; \
		if [ -n "$$PORT_PIDS" ]; then \
			for PID in $$PORT_PIDS; do \
				ARGS="$$(ps -p $$PID -o args= 2>/dev/null || true)"; \
				case "$$ARGS" in \
					*mlflow*|*fastapi_app:app*) PIDS="$$PIDS $$PID" ;; \
				esac; \
			done; \
			PIDS="$$(printf '%s\n' $$PIDS | awk 'NF' | sort -u | tr '\n' ' ')"; \
		fi; \
	fi; \
	if [ -n "$$PIDS" ]; then \
		echo "Stopping MLflow UI processes: $$PIDS"; \
		kill $$PIDS; \
	else \
		echo "No MLflow UI process found."; \
		if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$$PORT" -sTCP:LISTEN >/dev/null 2>&1; then \
			echo "Note: port $$PORT is occupied by a non-MLflow process:"; \
			lsof -nP -iTCP:"$$PORT" -sTCP:LISTEN; \
		fi; \
	fi
