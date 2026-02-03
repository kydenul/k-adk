# Go parameters
GOCMD := go
GOBUILD := $(GOCMD) build
GOCLEAN := $(GOCMD) clean
GOTEST := $(GOCMD) test
GOVET := $(GOCMD) vet
GOMOD := $(GOCMD) mod
GOFUMPT := gofumpt
GOLINT := golangci-lint

# Project parameters
PKG_LIST := $(shell $(GOCMD) list ./... | grep -v /vendor/ | grep -v /examples/)

# Debug configuration
DEBUG ?= false

# Colors for pretty printing
GREEN := \033[0;32m
BLUE := \033[0;34m
YELLOW := \033[0;33m
NC := \033[0m # No Color

# Targets
.PHONY: all build check test test-cover vet fumpt lint tidy clean help

# Default target
all: build

# Full build pipeline (for library: verify compilation + quality checks)
build: clean tidy fumpt lint check
	@printf "$(GREEN)✓ Build completed successfully!$(NC)\n"

# Quick compile without linting (verify library compiles)
compile: tidy check
	@printf "$(GREEN)✓ Compile completed!$(NC)\n"

# Verify library compiles correctly
check:
	@printf "$(BLUE)Checking library compilation ...$(NC)\n"
	@$(GOBUILD) ./...

# Run tests
test:
	@printf "$(BLUE)Running tests ...$(NC)\n"
	@$(GOTEST) -v $(PKG_LIST)

# Run tests with coverage
test-cover:
	@printf "$(BLUE)Running tests with coverage ...$(NC)\n"
	@$(GOTEST) -v -cover -coverprofile=coverage.out $(PKG_LIST)
	@$(GOCMD) tool cover -func=coverage.out
	@printf "$(GREEN)✓ Coverage report generated: coverage.out$(NC)\n"

# Run go vet
vet:
	@printf "$(BLUE)Running go vet ...$(NC)\n"
	@$(GOVET) ./...

# Format code with gofumpt
fumpt:
	@printf "$(BLUE)Running gofumpt ...$(NC)\n"
	@$(GOFUMPT) -w -l $(shell find . -name '*.go' -not -path './vendor/*')

# Run linter
lint: vet
	@printf "$(BLUE)Running linter ...$(NC)\n"
	@$(GOLINT) run ./...

# Tidy and verify module dependencies
tidy:
	@printf "$(BLUE)Tidying and verifying module dependencies ...$(NC)\n"
	@$(GOMOD) tidy
	@$(GOMOD) verify

# Clean build cache and generated files
clean:
	@printf "$(BLUE)Cleaning up ...$(NC)\n"
	@$(GOCLEAN)
	@rm -f coverage.out

# Display help
help:
	@echo "$(BLUE)K-ADK Library - Available targets:$(NC)"
	@echo ""
	@echo "  $(GREEN)all (build)$(NC)   : Full build pipeline (clean + tidy + fumpt + lint + check)"
	@echo "  $(GREEN)compile$(NC)       : Quick compile without linting (tidy + check)"
	@echo "  $(GREEN)check$(NC)         : Verify library compiles correctly"
	@echo "  $(GREEN)test$(NC)          : Run all tests with verbose output"
	@echo "  $(GREEN)test-cover$(NC)    : Run tests with coverage report"
	@echo "  $(GREEN)vet$(NC)           : Run go vet for static analysis"
	@echo "  $(GREEN)fumpt$(NC)         : Format code with gofumpt"
	@echo "  $(GREEN)lint$(NC)          : Run golangci-lint (includes vet)"
	@echo "  $(GREEN)tidy$(NC)          : Tidy and verify go modules"
	@echo "  $(GREEN)clean$(NC)         : Clean build cache and generated files"
	@echo "  $(GREEN)help$(NC)          : Display this help message"
	@echo ""
	@echo "$(BLUE)Examples:$(NC)"
	@echo "  make              # Full build pipeline"
	@echo "  make compile      # Quick compile without linting"
	@echo "  make test         # Run tests"
	@echo "  make test-cover   # Run tests with coverage"
	@echo "  make lint         # Run linter only"

# Debugging
print-%:
	@echo '$*=$($*)'
