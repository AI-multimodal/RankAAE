from setuptools import setup, find_packages

with open("requirements.txt") as f_req:
    required_list = [line.rstrip() for line in f_req.readlines()]

setup(
    name='Semi Clustering',
    version='0.1',
    packages=['sc'],
    url='https://www.bnl.gov/cfn/',
    license='GPL',
    author='Xiaohui Qu, Matthew Carbone',
    author_email='xiaqu@bnl.gov; mrc2215@columbia.edu',
    description='Semi-supervised Clustering',
    python_requires='>=3.7',
    install_requires=required_list,
    entry_points={
        "console_scripts": [
            "train_sc = sc.cmd.train_sc:main",
            "train_lat2apdf = sc.cmd.train_lat2apdf:main",
            "train_lat2prdf = sc.cmd.train_lat2prdf:main",
            "test_model = sc.cmd.test_model:main",
            "opt_hyper_single = sc.cmd.opt_hyper_single:main"
        ]
    },
    scripts=[
        "sc/cmd/opt_hyper.sh",
        "sc/cmd/train_on_worker"
    ]
)

