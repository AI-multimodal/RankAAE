#!/usr/bin/env bash

hn=$1

if [[ ${SLURM_LOCALID} == "0" ]]
then
    nvidia-smi -l 37 &> resources_usage_${SLURM_JOB_ID}/gpu_${SLURM_PROCID}_`hostname -s`.txt &
    top -i -b -d 31 &> resources_usage_${SLURM_JOB_ID}/cpu_${SLURM_PROCID}_`hostname -s`.txt &
fi

ipengine --profile-dir=ipypar --location=${hn} --log-to-file &
