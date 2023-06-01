#!/bin/bash

# Load secrets environment variables
source .env
docker-compose up -d

# Install local package askem
pip install .

# Run the deployment script
python ./deploy.py --init --input-dir "data/debug_data/" --topic "covid-19"
