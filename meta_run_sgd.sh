#!/bin/bash

py=/users/guillefix/anaconda3/envs/venv/bin/python
nprocs=1
train_size=20000
$py make_data.py --train_size $train_size
/users/guillefix/anaconda3/envs/venv/bin/mpiexec -n $nprocs $py sample_funs_sgd.py --train_size $train_size
