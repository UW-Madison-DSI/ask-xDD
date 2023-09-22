#!bin/bash

apt-get update
apt-get install -y locales
locale-gen en_US.UTF-8
pip install -r .devcontainer/requirements.txt
pip install -e .
pre-commit install
