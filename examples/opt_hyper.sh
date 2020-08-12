#!/usr/bin/env bash

num_jobs=$1
sh_fn=`realpath ${BASH_SOURCE}`
py_fn=${sh_fn/.sh/.py}

export CONDA_SHLVL=1
export CONDA_PROMPT_MODIFIER=(/hpcgpfs01/software/cfn-jupyter/software/xas_m    l)
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

nvidia-smi -l 37 &> ${res_dir}/gpu_${SLURM_PROCID}_`hostname -s`.txt &
top -i -b -d 31 &> ${res_dir}/cpu_${SLURM_PROCID}_`hostname -s`.txt &

log_dir=optuna_run_${SLURM_JOB_ID}
if [[ ! -d ${log_dir} ]]
then
    mkdir "${log_dir}"
fi

seq ${num_jobs} | parallel -j ${num_jobs} "
sleep \$(( ({#}-1) * 3 ))
python ${py_fn} -g \$(({%} % 4)) ${@:2} &> ${log_dir}/opt_${SLURM_PROCID}_`hostname -s`.txt
"
