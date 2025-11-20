#!/bin/bash
model=$(ollama ps | awk 'NR==2 {print $1}')
[ -n "$model" ] && curl -s http://localhost:11434/api/generate -d "{\"model\": \"$model\", \"keep_alive\": 0}"