#!/bin/bash

./build.sh 2.5.0 cu124
./build.sh 2.5.1 cu124

./build.sh 2.5.0 cu121
./build.sh 2.5.1 cu121

./build.sh 2.5.0 cu118
./build.sh 2.5.1 cu118

./build.sh 2.5.0 cpu
./build.sh 2.5.1 cpu

git restore setup.py
git restore src/torchCompactRadius/__init__.py

python genhtmls.py

git add .
git commit -m "update prebuilts"
git push