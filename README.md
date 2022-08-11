# semi_clustering
Cluster spectra with coarse grain constraint and disentangle chemical information automatically.

The clustering algorithm is based on the paper by P. Ge et al at IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS, 2020, 31(4), 1417.

# Installation

```bash
conda create -n test5 python=3.8 pytorch=1.9.0 cudatoolkit=11.1 numpy=1.21.1 torchvision tomli -c nvidia -c pytorch -c conda-forge

conda activate py37
pip install -U setuptools
git clone git@github.com:xhqu1981/semi_clustering.git
cd semi_clustering
python setup.py develop

### some other packages might be needed as well 
`optuna` needs two things to run properly: redis database and plotly
1. Follow https://redis.io/topics/quickstart to install redis-server
2. Use pip/conda to install node.js, npm, plotly 
3. Install plotly jupyterlab extension: `jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyterlab-plotly`


# Optional, for Jupyter notebooks
```

