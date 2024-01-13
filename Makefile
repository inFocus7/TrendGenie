TRENDGENIE_OUTPUT_DIR ?= ~/trendgenie
TRENDGENIE_SERVICE_NAME ?= trendgenie
TEARDOWN_PREVIOUS_CONTAINER ?= true

.PHONY: build
build:
	@echo "Building TrendGenie's Docker image..."
	docker build --no-cache -t $(TRENDGENIE_SERVICE_NAME) .
	@echo "TrendGenie's Docker image built successfully"

.PHONY: run
run:
	@echo "Running TrendGenie's Docker image..."
	docker run -d --name $(TRENDGENIE_SERVICE_NAME) -p 7860:7860 -v $(TRENDGENIE_OUTPUT_DIR):/root/trendgenie $(TRENDGENIE_SERVICE_NAME)
	@echo "TrendGenie is running on port 7860"

.PHONY: start
start:
	make build
	make run

.PHONY: stop
stop:
	@echo "Stopping TrendGenie's Docker container..."
	docker stop $(TRENDGENIE_SERVICE_NAME) || true
	@echo "TrendGenie's Docker container stopped successfully"

.PHONY: rm
rm:
	make stop
	@echo "Removing TrendGenie's Docker container..."
	docker rm $(TRENDGENIE_SERVICE_NAME) || true
	@echo "TrendGenie's Docker container removed successfully"

.PHONY: logs
logs:
	docker logs -f $(TRENDGENIE_SERVICE_NAME)
