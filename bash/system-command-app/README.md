# Personal AI Assistant - System Command Wrapper

To make the Personal AI Assistant easily accessible from anywhere in your terminal, you can create a symbolic link to the wrapper script.

## Prerequisites

The wrapper supports multiple Python environment types:

- **venv/uv**: Automatically detects and activates `.venv` in the project directory
- **conda**: Falls back to activating a conda environment named `mnemoai`

If using conda, create the environment first:

```bash
conda create -n mnemoai python=3.11
conda activate mnemoai
# Install dependencies...
```

## Installation

```bash
chmod +x bash/system-command-app/mnemoai-wrapper.sh
ln -sf $(pwd)/bash/system-command-app/mnemoai-wrapper.sh /usr/local/bin/mnemoai
```

## Usage

Then run from anywhere:

```bash
mnemoai
```
