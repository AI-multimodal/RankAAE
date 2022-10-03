# RankAAE
An neural network based algorithm that constructs a latent space in line with physical descriptors for XANES spectra.

The clustering algorithm is based on the paper by P. Ge et al at IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, 2020, 31(4), 1417.

# Installation

It is highly recommended to use a `conda` vertual environment for the package and all its dependencies.

```bash
conda create -n your_env_name python=3.8 pytorch=1.9.0 cudatoolkit=11.1 numpy=1.21.1 torchvision tomli -c nvidia -c pytorch -c conda-forge

conda activate your_env_name

pip install -U setuptools
```

Get the code using git and install the package in development mode for easy modification
```bash
git clone git@github.com:xhqu1981/RankAAE.git
cd RankAAE
python setup.py develop
```

### some other packages might be needed as well 
1. Use pip/conda to install node.js, npm, plotly 
2. Install plotly jupyterlab extension: `jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyterlab-plotly`

# Usage
In the "example" folder, you will see three files: the execution bash script, the configuration file and the data file. For a simple demo training, simply locate to that folder and execute `run_training.sh`.
The `fix_config.yaml` file controls all the configuration parameters, including the data file, training rounds, model configuration, reporting spcification, etc.
Note that: if `trials` is set to `1` in configuration, no need to start ipyparallel engine, otherwise ipyparallel will be started automatically.