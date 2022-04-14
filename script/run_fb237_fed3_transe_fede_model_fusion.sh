#!/bin/sh

gpu=0

python main.py --data_path ./data/FB15k237-Fed3.pkl --name fb15k237_fed3_transe_fede_model_fusion \
              --setting Model_Fusion --mode train --model TransE --gpu $gpu \
              --fusion_state fb15k237_fed3_transe_isolation fb15k237_fed3_transe_fede