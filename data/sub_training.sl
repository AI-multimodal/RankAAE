#!/bin/bash
#SBATCH -p long
#SBATCH -J mutlinode
#SBATCH --time=1-00:00:00
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH -c 36
#SBATCH --overcommit

total_jobs=100

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

echo create profile at `date`
echo "Creating profile "
ipython profile create --profile-dir=ipypar
 
echo "Launching controller"
ipcontroller --ip="*" --profile-dir=ipypar --log-to-file &
sleep  60
  
echo "Launching engines"
hn=$(hostname -s)
srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node ${num_jobs} start_ipyparallel_worker.sh ${hn} &
sleep  240

echo `date` "Start training"
train_sc --trials ${total_jobs} -c fix_config.yaml -d cu_feff_cn_wei_spec.csv -e 2000 -v
echo `date` "Job Finished"

