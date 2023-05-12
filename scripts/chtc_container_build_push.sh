#!/bin/bash

# Build the container
docker build . -f chtc/Dockerfile -t ghcr.io/jasonlo/askem-chtc

# Push the container
docker push ghcr.io/jasonlo/askem-chtc
