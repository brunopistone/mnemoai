# Ollama VRAM Cleaner

Script and service to periodically free up VRAM used by Ollama on Linux and macOS.

--

### Step 1:

Replace `/PATH_TO_REPOSITORY` and `/PATH_TO_USER_HOME` with your actual paths.

### Step 2:

Run `chmod +x ollama_free_vram.sh`

## Linux

### Step 3:

Install the service and its timer:

```bash
sudo cp ollama-vram-cleaner.service ollama-vram-cleaner.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ollama-vram-cleaner.timer
```

The timer runs the one-shot service every minute.

## Mac

### Step 3:

Copy `com.ollama.vramcleaner.plist` under `/Library/LaunchDaemons/`

```
sudo cp com.ollama.vramcleaner.plist /Library/LaunchDaemons/
```

### Step 4:

Set permissions:

`sudo chmod 644 /Library/LaunchDaemons/com.ollama.vramcleaner.plist`

### Step 4:

Launch and reload daemons:

```
sudo launchctl load /Library/LaunchDaemons/com.ollama.vramcleaner.plist
sudo launchctl start com.ollama.vramcleaner
```
