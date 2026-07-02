.PHONY: doctor clean worker-setup docker-doctor docker-build docker-up docker-up-gpu docker-worker docker-worker-gpu docker-down docker-logs docker-ps docker-train-help

doctor:
	python scripts/system_check.py

clean:
	find . -path './.venv' -prune -o \( -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -path './data/.cache' \) -exec rm -rf {} + \)
	rm -f chat-server.log server-local-worker.log data/chat_live/learned.jsonl.bak-*

worker-setup:
	python scripts/setup_worker.py --server "$${SERVER:-http://127.0.0.1:8000}" --token "$${TOKEN:-$${YERELLM_API_TOKEN:-}}" --name "$${NAME:-$$(hostname)}"

docker-doctor:
	docker compose --profile doctor run --rm doctor

docker-build:
	docker compose build api

docker-up:
	docker compose up -d api

docker-up-gpu:
	docker compose --profile gpu up -d --build api-gpu

docker-worker:
	docker compose --profile worker up -d worker

docker-worker-gpu:
	docker compose --profile gpu --profile gpu-worker up -d worker-gpu

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-ps:
	docker compose ps

docker-train-help:
	docker compose --profile train run --rm trainer train --help
