# Makefile for Llama Stack

# Configuration file
CONFIG_FILE = run.yaml

# Port
PORT = 8321

# Environment variables
OLLAMA_URL = http://localhost:11434
INFERENCE_MODEL = llama4
SQLITE_STORE_DIR = ~/.llama/distributions/ollama

# Targets
.PHONY: run-ollama run-llamastack tail-ollama-logs curl-ollama run-all stop-llamastack

run-ollama:
	@ollama run llama3.2:1b-instruct-fp16 >/dev/null 2>&1 &
	@echo "ollama running"

run-llamastack:
	docker run \
		-d \
		--name llamastack \
		-it \
		-p $(PORT):$(PORT) \
		-v ~/.llama:/root/.llama \
		-v ./$(CONFIG_FILE):/root/$(CONFIG_FILE) \
		-e INFERENCE_MODEL=meta-llama/Llama-3.2-1B-Instruct \
		-e OLLAMA_URL=http://host.docker.internal:11434 \
		llamastack/distribution-ollama \
		--yaml-config /root/$(CONFIG_FILE) \
		--port $(PORT)

run-all: run-ollama
	@sleep 3  # wait a bit to let ollama start
	@$(MAKE) run-llamastack
	@echo "Both ollama and llamastack are running."

tail-ollama-logs:
	tail -f ~/.ollama/logs/server.log

curl-ollama:
	curl http://localhost:11434/api/generate -d '{ "model": "llama3.2:1b-instruct-fp16", "prompt": "Hello!", "stream": false }'

stop-llamastack:
	@docker stop llamastack
	@docker rm llamastack


# should have 2 model 1 for embedding(all-minilm:latest) , 1 for inference(llama3.2:1b-instruct-fp16)
