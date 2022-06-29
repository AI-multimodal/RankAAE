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

total_engines=16
total_jobs=16

element=$1
work_dir="../tests/work_dir_$1"
cd ${work_dir}
data_file=$(ls feff_$1_*)
echo ${data_file}
rm -r training

ipython profile create --profile-dir=ipypar
ipcontroller --ip="*" --profile-dir=ipypar &
sleep 10

for i in `seq 1 $total_engines`
do
export SLURM_LOCALID=$i
ipengine --profile-dir=ipypar --log-to-file &
done

wait_ipp_engines -e $total_engines
echo "Engines seems to have started"

echo `date` "Start training"
train_sc \
    -d ${data_file} \
    --trials ${total_jobs} \
    -c fix_config.yaml \
    -e 1500 \
    -v
echo `date` "Job Finished"
stop_ipcontroller
rm -r ipypar

echo `date` "Genearting Report"
current_folder=$(basename `pwd`)
parent_folder=$(basename $(dirname `pwd`))
sc_generate_report \
    -o report_${parent_folder}_${current_folder} \
    -n 5 \
    -p 5 \
    -g 
echo `date` "Report Generated"

