# =============================================================================
# memill - Makefile
# =============================================================================
# Usage: make <target> [urls ...]
# Run 'make help' to see all available commands
#
# URL convention:
#   Extra words on the command line are URLs, not targets. A catch-all rule
#   swallows them so make does not try to build them, and commas between URLs
#   are stripped, so all of these work:
#
#     make dw https://youtu.be/AAA
#     make dw https://youtu.be/AAA, https://youtu.be/BBB
#     make dw-ff urls.txt
#
# QUOTE ANY URL CONTAINING '&'. Your shell splits on '&' before make ever runs,
# so a playlist link like ...?v=AAA&list=BBB must be quoted or it silently
# becomes two commands. Short youtu.be/ID links never need quoting.
#
# The catch-all also means a mistyped target does nothing instead of erroring.
# That is the price of unquoted URLs; 'make help' lists what is real.
# =============================================================================

.PHONY: help \
        dw dw-ff dry dry-ff \
        install uninstall \
        test lint types check clean

.DEFAULT_GOAL := help

# Recipes run in bash so 'echo' interprets the \033 colour escapes (dash does not).
SHELL := /bin/bash
.SHELLFLAGS := -O xpg_echo -c

# --- Variables ---------------------------------------------------------------
VENV            ?= .venv
YT2MP3          ?= $(VENV)/bin/memill
PYTEST          ?= $(VENV)/bin/pytest
RUFF            ?= $(VENV)/bin/ruff
MYPY            ?= $(VENV)/bin/mypy

# Passed straight through to memill, e.g. make dw URL FLAGS="--normalize -j 8"
FLAGS           ?=

# --- URL capture -------------------------------------------------------------
# Every command-line word after the target is treated as a URL. Commas between
# them are stripped so 'make dw A, B' reads the same as 'make dw A B'.
COMMA           := ,
EMPTY           :=
SPACE           := $(EMPTY) $(EMPTY)
ARGS            := $(filter-out $(firstword $(MAKECMDGOALS)),$(MAKECMDGOALS))
URLS            := $(strip $(subst $(COMMA),$(SPACE),$(ARGS)))

# --- Colors ------------------------------------------------------------------
CYAN   := \033[36m
GREEN  := \033[32m
YELLOW := \033[33m
RESET  := \033[0m

# =============================================================================
# HELP
# =============================================================================
help: ## Show this help message
	@echo ""
	@echo "$(CYAN)memill - download YouTube audio as tagged MP3$(RESET)"
	@echo ""
	@echo "  URLs go straight after the target: $(CYAN)make dw https://youtu.be/AAA, https://youtu.be/BBB$(RESET)"
	@echo "  $(YELLOW)Quote any URL containing '&' - your shell splits on it before make runs.$(RESET)"
	@echo "  Extra flags: $(CYAN)make dw <url> FLAGS=\"--normalize -j 8\"$(RESET)"
	@echo ""
	@echo "$(GREEN)Download:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(dw|dw-ff|dry|dry-ff):' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)make %-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Install:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(install|uninstall):' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)make %-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Code Quality:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(test|lint|types|check|clean):' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)make %-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# DOWNLOAD
# =============================================================================
dw: ## Download one or more URLs. Usage: make dw <url> [url ...]
	@test -n "$(URLS)" || { echo "$(YELLOW)Usage: make dw <url> [url ...]$(RESET)"; exit 1; }
	@test -x $(YT2MP3) || { echo "$(YELLOW)$(YT2MP3) not found - run 'make install' first$(RESET)"; exit 1; }
	$(YT2MP3) $(FLAGS) $(URLS)

dw-ff: ## Download every URL in a file. Usage: make dw-ff <file>
	@test -n "$(URLS)" || { echo "$(YELLOW)Usage: make dw-ff <file>$(RESET)"; exit 1; }
	@test -f "$(firstword $(URLS))" || { echo "$(YELLOW)No such file: $(firstword $(URLS))$(RESET)"; exit 1; }
	$(YT2MP3) $(FLAGS) --from-file "$(firstword $(URLS))"

dry: ## List what would be fetched, download nothing. Usage: make dry <url> [url ...]
	@test -n "$(URLS)" || { echo "$(YELLOW)Usage: make dry <url> [url ...]$(RESET)"; exit 1; }
	$(YT2MP3) --dry-run $(FLAGS) $(URLS)

dry-ff: ## List what a file would fetch. Usage: make dry-ff <file>
	@test -n "$(URLS)" || { echo "$(YELLOW)Usage: make dry-ff <file>$(RESET)"; exit 1; }
	$(YT2MP3) --dry-run $(FLAGS) --from-file "$(firstword $(URLS))"

# =============================================================================
# INSTALL
# =============================================================================
install: ## Create the venv, install the package, link memill onto PATH
	@./scripts/dev-setup.sh

uninstall: ## Remove the ~/.local/bin/memill launcher (leaves the venv alone)
	@if [ -L "$$HOME/.local/bin/memill" ]; then \
		rm -- "$$HOME/.local/bin/memill"; \
		echo "$(GREEN)Removed $$HOME/.local/bin/memill$(RESET)"; \
	else \
		echo "$(YELLOW)Nothing to remove at $$HOME/.local/bin/memill$(RESET)"; \
	fi

# =============================================================================
# CODE QUALITY
# =============================================================================
test: ## Run the test suite
	$(PYTEST)

lint: ## Run ruff
	$(RUFF) check .

types: ## Run mypy over src
	$(MYPY) src

check: lint types test ## Run ruff, mypy and the suite

clean: ## Remove caches and stale staging directories
	@rm -rf .pytest_cache .mypy_cache .ruff_cache
	@find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@rm -rf "$$HOME/.cache/memill"
	@echo "$(GREEN)Cleaned caches and $$HOME/.cache/memill$(RESET)"

# =============================================================================
# Swallow the URL words so make does not try to build them as targets.
# Keep this last; it is why a mistyped target fails silently.
# =============================================================================
%:
	@:
