# Ollama VRAM Cleaner

Script and service to periodically free up VRAM used by Ollama on Linux and macOS.

--

### Step 1:

Edit `<PATH_TO_FOLDER>` and `<PATH_TO_USER_HOME>` from the files

### Step 2:

Run `chmod +x ollama_free_vram.sh`

## Linux

### Step 3:

Copy `ollama-vram-cleaner.service` under `/etc/systemd/system/`

`cp ollama-vram-cleaner.service /etc/systemd/system/`

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
