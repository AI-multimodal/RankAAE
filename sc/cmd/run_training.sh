#!/bin/bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
export MKL_DOMAIN_NUM_THREADS="MKL_DOMAIN_ALL=1, MKL_DOMAIN_BLAS=1"
export MKL_NUM_STRIPES=1
export NUMEXPR_NUM_THREADS=1

ulimit -u 524288
ulimit -n 100000
ulimit -a

ipcluster start -n=8 --profile-dir=ipypar &
sleep 10
echo `date` "Start training"
train_sc -c fix_config.yaml
echo `date` "Job Finished"
ipcluster stop --profile-dir=ipypar

echo `date` "Genearting Report"
sc_generate_report -c fix_config.yaml
echo `date` "Report Generated"

