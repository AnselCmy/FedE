#!/bin/sh

gpu=1

python main.py --data_path ./data/FB15k237-Fed3.pkl --name fb15k237_fed3_transe_collection \
              --setting Collection --mode train --model TransE --gpu $gpu