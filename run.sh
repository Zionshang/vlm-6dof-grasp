#!/bin/bash
# export PYTHONPATH=/home/jetson/python_ws/arx5-sdk/python:$PYTHONPATH
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export PYTHONPATH=/home/jyx/python_ws/arx5-sdk/python:$PYTHONPATH
python run_realtime.py "$@"
