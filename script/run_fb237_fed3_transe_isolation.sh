#!/bin/sh

gpu=0

for client in 0 1 2;
do
  python main.py --data_path ./data/FB15k237-Fed3.pkl --name fb15k237_fed3_transe_isolation \
              --setting Isolation --one_client_idx $client \
              --mode train --model TransE --gpu $gpu
done