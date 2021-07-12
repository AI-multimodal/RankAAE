#!/usr/bin/env bash

worker=`cat hostfile | head -n $1 | tail -n 1`
srun -N 1 -n 1 -w $worker train_sc ${@:2}
