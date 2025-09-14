# Makefile for GAN Introduction Slides

# Variables
SLIDES_DIR = doc/slides
SLIDES_FILE = gan_introduction.qmd
OUTPUT_DIR = $(SLIDES_DIR)/output

# Default target
.PHONY: all
all: slides

# Build the slides
.PHONY: slides
slides:
	@echo "Building Quarto presentation..."
	@mkdir -p $(OUTPUT_DIR)
	@if [ ! -d "gan-env" ]; then echo "Virtual environment not found. Run 'make install-deps' first."; exit 1; fi
	cd $(SLIDES_DIR) && QUARTO_PYTHON=../../gan-env/bin/python quarto render $(SLIDES_FILE) --to revealjs
	@echo "Slides built successfully!"

# Preview the slides (opens in browser)
.PHONY: preview
preview:
	@echo "Starting Quarto preview..."
	@if [ ! -d "gan-env" ]; then echo "Virtual environment not found. Run 'make install-deps' first."; exit 1; fi
	cd $(SLIDES_DIR) && QUARTO_PYTHON=../../gan-env/bin/python quarto preview $(SLIDES_FILE) --to revealjs

# Clean output files
.PHONY: clean
clean:
	@echo "Cleaning output files..."
	rm -rf $(OUTPUT_DIR)
	rm -f $(SLIDES_DIR)/*.html
	rm -rf $(SLIDES_DIR)/*_files
	@echo "Clean completed!"

# Clean everything including virtual environment
.PHONY: clean-all
clean-all: clean
	@echo "Removing virtual environment..."
	rm -rf gan-env
	@echo "Full clean completed!"

# Install Quarto (macOS with Homebrew)
.PHONY: install-quarto
install-quarto:
	@echo "Installing Quarto via Homebrew..."
	brew install quarto

# Install Python dependencies in virtual environment
.PHONY: install-deps
install-deps:
	@echo "Creating virtual environment 'gan-env'..."
	python3 -m venv gan-env
	@echo "Installing Python dependencies in virtual environment..."
	./gan-env/bin/pip install --upgrade pip
	./gan-env/bin/pip install matplotlib numpy scipy torch jupyter

# Activate virtual environment (show instructions)
.PHONY: activate
activate:
	@echo "To activate the virtual environment, run:"
	@echo "source gan-env/bin/activate"
	@echo ""
	@echo "To deactivate when done, run:"
	@echo "deactivate"

# Help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  slides       - Build the presentation"
	@echo "  preview      - Preview slides in browser"
	@echo "  clean        - Remove output files"
	@echo "  clean-all    - Remove output files and virtual environment"
	@echo "  install-quarto - Install Quarto via Homebrew"
	@echo "  install-deps - Create virtual environment and install Python dependencies"
	@echo "  activate     - Show instructions to activate virtual environment"
	@echo "  help         - Show this help message"
