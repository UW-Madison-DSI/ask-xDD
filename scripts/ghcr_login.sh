#!/bin/bash

# Get secrets from .env
export $(grep -v '^#' .env | xargs -d '\n')

# Login to GitHub Container Registry
docker login ghcr.io -u JasonLo -p $GHCR_TOKEN
