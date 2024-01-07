TRENDGENIE_OUTPUT_DIR ?= ~/trendgenie/images
TRENDGENIE_SERVICE_NAME ?= trendgenie

.PHONY: build
build:
	@echo "Building TrendGenie's Docker image..."
	docker build -t $(TRENDGENIE_SERVICE_NAME) .
	@echo "TrendGenie's Docker image built successfully"

.PHONY: run
run:
	@echo "Running TrendGenie's Docker image..."
	docker run -d --name $(TRENDGENIE_SERVICE_NAME) -p 7860:7860 -v $(TRENDGENIE_OUTPUT_DIR):/trendgenie/images $(TRENDGENIE_SERVICE_NAME)
	@echo "TrendGenie is running on port 7860"

.PHONY: start
start:
	make build
	make run

.PHONY: logs
logs:
	docker logs -f $(TRENDGENIE_SERVICE_NAME)
