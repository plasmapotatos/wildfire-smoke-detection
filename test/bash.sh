#!/bin/bash

source /home/wei/miniconda3/bin/activate wildfire-smoke-dataset
python agents/paligemma_agent.py
bash ../shutdown.sh