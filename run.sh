#!/bin/bash
export PYTHONPATH=/home/jetson/python_ws/arx5-sdk/python:$PYTHONPATH
# export PYTHONPATH=/home/jyx/python_ws/arx5-sdk/python:$PYTHONPATH
python run_realtime.py "$@"
