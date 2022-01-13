#!/bin/bash

# Absolute path for root directory to the project
root_path=$(pwd)

. venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$root_path