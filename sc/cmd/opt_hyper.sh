#!/usr/bin/env bash

num_jobs=$1
head_node=$2   ## can be obtained from ${SLURMD_NODENAME}
sh_fn=$(realpath "${BASH_SOURCE}")
py_fn=${sh_fn/.sh/.py}

gpu_type=$(nvidia-smi | head -n 8 | tail -n 1 | tr -s ' ' | cut -d ' ' -f 4)
gpu_type=${gpu_type%-*}

if [[ ${gpu_type} == "P100" ]]
then
    num_gpus=2
elif [[ ${gpu_type} == "K80" ]]
then
    num_gpus=4
else
    num_gpus=-1
fi

export CONDA_SHLVL=1
export CONDA_PROMPT_MODIFIER=(/hpcgpfs01/software/cfn-jupyter/software/xas_ml)
export CONDA_EXE=/sdcc/u/xiaqu/program/miniconda3/bin/conda
export PATH=/hpcgpfs01/software/cfn-jupyter/software/xas_ml/bin:$PATH
export CONDA_PREFIX=/hpcgpfs01/software/cfn-jupyter/software/xas_ml
export CONDA_PYTHON_EXE=/sdcc/u/xiaqu/program/miniconda3/bin/python
export CONDA_DEFAULT_ENV=/hpcgpfs01/software/cfn-jupyter/software/xas_ml

res_dir=resources_usage_${SLURM_JOB_ID}
if [[ ! -d ${res_dir} ]]
then
    mkdir "${res_dir}"
fi

hn=$(hostname -s)
nvidia-smi -l 37 &> ${res_dir}/gpu_${SLURM_PROCID}_${hn}.txt &
top -i -b -d 31 &> ${res_dir}/cpu_${SLURM_PROCID}_${hn}.txt &

if [[ ${hn} == "${head_node}" ]]
then
    echo Land on head node, start Redis
    sed -i "s/^bind.*/bind ${hn}/" redis.conf
    redis-server redis.conf &
fi

log_dir=optuna_run_${SLURM_JOB_ID}
if [[ ! -d ${log_dir} ]]
then
    mkdir "${log_dir}"
fi

seq ${num_jobs} | parallel -j ${num_jobs} "
sleep \$(( ({#}-1) * 3 + (${SLURM_PROCID} * ${num_jobs} + 1) * 3 ))
python ${py_fn} -g \$(({%} % ${num_gpus})) -a ${head_node} ${@:3} &> ${log_dir}/opt_${SLURM_PROCID}_{#}_${hn}.txt
"

redis-cli -h ${hn} shutdown
