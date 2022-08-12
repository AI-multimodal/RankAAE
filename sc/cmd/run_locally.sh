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
total_jobs=100

element=$1
work_dir="work_dir_$1"
data_file=$(ls feff_$1_*)
echo ${data_file}
rm -r training

ipython profile create --profile-dir=${work_dir}/ipypar
ipcontroller --ip="*" --profile-dir=${work_dir}/ipypar &
sleep 10

for i in `seq 1 $total_engines`
do
    export SLURM_LOCALID=$i
    ipengine --profile-dir=${work_dir}/ipypar --log-to-file &
done

wait_ipp_engines -w ${work_dir} -e $total_engines
echo "Engines seems to have started"

echo `date` "Start training"
echo train_sc -w ${work_dir} -c fix_config.yaml
train_sc -w ${work_dir} -c fix_config.yaml
echo `date` "Job Finished"
stop_ipcontroller
rm -r ${work_dir}/ipypar

echo `date` "Genearting Report"
current_folder=$(basename `pwd`)
parent_folder=$(basename $(dirname `pwd`))
echo sc_generate_report -w ${work_dir} -c fix_config.yaml
sc_generate_report -w ${work_dir} -c fix_config.yaml
echo `date` "Report Generated"

