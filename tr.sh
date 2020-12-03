#!/usr/bin/env bash
source venv/bin/activate
python3 dataset.py --train_path 10movies_max4_tag.train.json --dev_path 10movies_max4_tag.dev.json --batch_size 40
