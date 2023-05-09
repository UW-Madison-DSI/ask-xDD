#!/bin/bash

source .env
echo "Loaded APIKEY: $WEAVIATE_APIKEY"
docker-compose up -d
