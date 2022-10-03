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

<<<<<<< HEAD
# Usage
In the "example" folder, you will see three files: the execution bash script, the configuration file and the data file. For a simple demo training, simply locate to that folder and execute `run_training.sh`.
The `fix_config.yaml` file controls all the configuration parameters, including the data file, training rounds, model configuration, reporting spcification, etc.
Note that: if `trials` is set to `1` in configuration, no need to start ipyparallel engine, otherwise ipyparallel will be started automatically.
=======
# Funding acknowledgement

This research is based upon work supported by the U.S. Department of Energy, Office of Science, Office Basic Energy Sciences, under Award Number FWP PS-030. This research used resources of the Center for Functional Nanomaterials (CFN), which is a U.S. Department of Energy Office of Science User Facility, at Brookhaven National Laboratory under Contract No. DE-SC0012704.

## Disclaimer

The Software resulted from work developed under a U.S. Government Contract No. DE-SC0012704 and are subject to the following terms: the U.S. Government is granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable worldwide license in this computer software and data to reproduce, prepare derivative works, and perform publicly and display publicly.

THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND. THE UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, AND THEIR EMPLOYEES: (1) DISCLAIM ANY WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT, (2) DO NOT ASSUME ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF THE SOFTWARE, (3) DO NOT REPRESENT THAT USE OF THE SOFTWARE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS, (4) DO NOT WARRANT THAT THE SOFTWARE WILL FUNCTION UNINTERRUPTED, THAT IT IS ERROR-FREE OR THAT ANY ERRORS WILL BE CORRECTED.

IN NO EVENT SHALL THE UNITED STATES, THE UNITED STATES DEPARTMENT OF ENERGY, OR THEIR EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, CONSEQUENTIAL, SPECIAL OR PUNITIVE DAMAGES OF ANY KIND OR NATURE RESULTING FROM EXERCISE OF THIS LICENSE AGREEMENT OR THE USE OF THE SOFTWARE.
>>>>>>> b4a336d9e19534d5f1c77383b2029b689a48c862
