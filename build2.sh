#!/bin/bash

./build.sh 2.7.1 cu118

./build.sh 2.7.1 cu126

./build.sh 2.7.1 cu128

./build.sh 2.7.1 cpu

git restore setup.py
git restore src/torchCompactRadius/__init__.py

python genhtmls.py

git add .
git commit -m "update prebuilts"
git push