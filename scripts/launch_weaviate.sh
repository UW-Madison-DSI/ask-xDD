#!/bin/bash

# Load secrets environment variables
source .env
docker-compose up -d

# Install local package askem
pip install -e .

# Run the deployment script
python ./askem/deploy.py -f --input-dir "/askem/data/debug_data/"
