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

total_jobs=8

ipcluster start -n=8 --profile-dir=ipypar &
sleep 40

echo `date` "Start training"
train_sc \
    -d feff_Co_CT_CN_OCN_RSTD_MOOD_spec_202201161134_4000.csv \
    --trials ${total_jobs} \
    -c fix_config.yaml \
    -e 1500 \
    -v
echo `date` "Job Finished"

echo `date` "Genearting Report"
current_folder=$(basename `pwd`)
parent_folder=$(basename $(dirname `pwd`))
python ~/Documents/semi_clustering/sc/report/generate_report.py \
    -o report_${parent_folder}_${current_folder} \
    -n 5 \
    -p ${total_jobs} \
    -g 
echo `date` "Report Generated"

ipcluster stop --profile-dir=ipypar