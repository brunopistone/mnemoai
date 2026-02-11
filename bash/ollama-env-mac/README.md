# Ollama Configuration

## Setup

This configures Ollama to run with optimized settings on macOS.

### Environment Variables

- `OLLAMA_FLASH_ATTENTION=1` - Enables flash attention for faster inference
- `OLLAMA_KV_CACHE_TYPE=q8_0` - Uses 8-bit float precision for KV cache

### Installation

1. Copy the plist file to LaunchAgents:

```bash
cp ollama.environment.plist ~/Library/LaunchAgents/
```

2. Load the launch agent:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ollama.environment.plist
```

3. Restart Ollama:

```bash
killall Ollama ollama
open -a Ollama
```

### Verify

Check that environment variables are set:

```bash
ps aux | grep "ollama serve" | grep -v grep | awk '{print $2}' | head -1 | xargs ps eww -p | tr ' ' '\n' | grep OLLAMA
```

### Modify Settings

Edit `~/Library/LaunchAgents/ollama.environment.plist` and reload:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/ollama.environment.plist 2>/dev/null
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ollama.environment.plist
```

> Note: The `bootout` command may fail with an I/O error if the agent has already finished running. This is expected since the agent uses `LaunchOnlyOnce` and can be safely ignored.

### KV Cache Options

- `f16` - 16-bit float (default, highest quality, most memory)
- `q8_0` - 8-bit quantization (~50% memory savings)
- `q4_0` - 4-bit quantization (maximum memory savings)
