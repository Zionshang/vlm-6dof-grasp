#!/bin/bash
export PYTHONPATH=/home/jetson/python_ws/arx5-sdk/python:$PYTHONPATH
python run_realtime.py "$@"
