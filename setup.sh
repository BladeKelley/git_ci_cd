#! /bin/bash

set -e

echo 'remove .venv/'
rm -rf .venv/

echo 'create fresh .venv'
python3.8 -m venv .venv

echo 'activate'
source ./.venv/bin/activate

echo 'upgrade pip'
pip install --upgrade pip

echo 'pip install'
pip install -r ./dev/requirements.txt

echo 'Make folder structure'
mkdir -p .data
mkdir -p .model
