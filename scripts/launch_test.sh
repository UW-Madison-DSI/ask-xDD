#!/bin/bash

# Load secrets environment variables
source .env
docker-compose up -d

# Install local package askem
pip install .

# Run weaviate init (Do not init if already initialized, it will wipe everything!!!)
python3 askem/init_class.py

# Run weaviate ingest script to ingest all data in `data/debug_data/`
python3 askem/ingest_docs.py --input-dir "data/debug_data/" --topic "covid-19" --doc-type "paragraph"

# Optionally explore weaviate python client here:  `notebooks/weaviate_query_example.ipynb`
