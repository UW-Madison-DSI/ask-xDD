#!/bin/bash

# This will resume from ./tmp/id2topics.pkl
# nohup python ./askem/ingest_v2.py --resume &> ingest.log &

# This will start from scratch
cd /hdd/clo36/repo/ask-xDD
source venv/bin/activate
nohup python ./askem/ingest_v2.py &> "tmp/ingest.log" &