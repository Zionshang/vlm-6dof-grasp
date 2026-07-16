#!/bin/bash
sudo route add -net 224.0.0.0 netmask 240.0.0.0 dev eno1
export PYTHONPATH=/home/jetson/python_ws/arx5-sdk/python:$PYTHONPATH
# export PYTHONPATH=/home/jyx/python_ws/arx5-sdk/python:$PYTHONPATH
python apps/run_grasp_lcm.py "$@"
