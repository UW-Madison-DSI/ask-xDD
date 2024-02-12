#!/bin/bash

current_date=$(date +"%Y-%m-%d")

# This will resume from ./tmp/id2topics.pkl
# nohup python ./askem/ingest_v2.py --resume &> ingest.log &

# This will start from scratch
nohup python ./askem/ingest_v2.py &> "tmp/ingest_$current_date.log" &