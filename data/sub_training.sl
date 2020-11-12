#!/bin/bash
#SBATCH -p long
#SBATCH -J test50
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH -c 36

total_jobs=50

export CONDA_SHLVL=1
export CONDA_PROMPT_MODIFIER=(/hpcgpfs01/software/cfn-jupyter/software/xas_ml)
export CONDA_EXE=/sdcc/u/xiaqu/program/miniconda3/bin/conda
export PATH=/hpcgpfs01/software/cfn-jupyter/software/xas_ml/bin:$PATH
export CONDA_PREFIX=/hpcgpfs01/software/cfn-jupyter/software/xas_ml
export CONDA_PYTHON_EXE=/sdcc/u/xiaqu/program/miniconda3/bin/python
export CONDA_DEFAULT_ENV=/hpcgpfs01/software/cfn-jupyter/software/xas_ml

gpu_type=$(nvidia-smi | grep Tesla | head -n 1 | tr -s ' ' | cut -d ' ' -f 4)
gpu_type=${gpu_type%-*}

echo $gpu_type

if [[ ${gpu_type} == "P100" ]]
then
    num_gpus=2
elif [[ ${gpu_type} == "K80" ]]
then
    num_gpus=4
else
    num_gpus=-1
fi

num_jobs=$((num_gpus * 2))

res_dir=resources_usage_${SLURM_JOB_ID}
if [[ ! -d ${res_dir} ]]
then
    mkdir "${res_dir}"
fi

hn=$(hostname -s)
nvidia-smi -l 37 &> ${res_dir}/gpu_${SLURM_PROCID}_${hn}.txt &
top -i -b -d 31 &> ${res_dir}/cpu_${SLURM_PROCID}_${hn}.txt &

log_dir=jobs_${SLURM_JOB_ID}
if [[ ! -d ${log_dir} ]]
then
    mkdir "${log_dir}"
fi

seq ${total_jobs} | parallel -j ${num_jobs} "
sleep \$(( ({#}-1) * 3 + (${SLURM_PROCID} * ${num_jobs} + 1) * 3 ))
train_sc -g \$(({%} % ${num_gpus})) -w training/job_{#} -c fix_config.yaml -d ti_feff_cn_spec.csv -v &> ${log_dir}/job_${SLURM_PROCID}_{#}_${hn}.txt
"
