#!/bin/bash
export $(grep -v '^#' .env | xargs -d '\n') && docker login ghcr.io -u JasonLo -p $GHCR_TOKEN