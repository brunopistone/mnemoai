# Ollama Utilities (Optional)

The `bash/` directory contains helper scripts for Ollama users on macOS and Linux.

### Ollama Environment Setup (macOS)

Sets Ollama performance environment variables at boot and launches the Ollama app:

```bash
# Variables set: OLLAMA_FLASH_ATTENTION=1, OLLAMA_KV_CACHE_TYPE=q8_0, OLLAMA_NUM_GPU=999
```

**Setup:**

1. Edit `bash/ollama-env-mac/ollama.environment.plist` (no changes needed for defaults)
2. Copy to LaunchAgents:

```bash
cp bash/ollama-env-mac/ollama.environment.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/ollama.environment.plist
```

### VRAM Cleaner

Automatically unloads idle Ollama models from VRAM to free GPU memory. Useful when running multiple models or when GPU memory is limited.

**macOS (LaunchAgent, runs every 60 seconds):**

1. Edit `bash/ollama-freeup-vram/com.ollama.vramcleaner.plist`:
   - Replace `<PATH_TO_FOLDER>` with the actual path to this repository
   - Replace `<PATH_TO_USER_HOME>` with your home directory
2. Install:

```bash
cp bash/ollama-freeup-vram/com.ollama.vramcleaner.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.ollama.vramcleaner.plist
```

**Linux (systemd):**

1. Edit `bash/ollama-freeup-vram/ollama-vram-cleaner.service`:
   - Replace `<PATH_TO_FOLDER>` with the actual path
2. Install:

```bash
sudo cp bash/ollama-freeup-vram/ollama-vram-cleaner.service /etc/systemd/system/
sudo systemctl enable ollama-vram-cleaner
sudo systemctl start ollama-vram-cleaner
```

See `bash/ollama-freeup-vram/README.md` and `bash/ollama-env-mac/README.md` for more details.
