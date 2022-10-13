from setuptools import setup, find_packages

with open("requirements.txt") as f_req:
    required_list = [line.rstrip() for line in f_req.readlines()]

setup(
    name='RankAAE',
    version='0.1',
    packages=['sc'],
    url='https://www.bnl.gov/cfn/',
    license='GPL',
    author='Xiaohui Qu, Zhu Liang',
    author_email='xiaqu@bnl.gov; zliang@bnl.gov',
    description='Semi-supervised Clustering',
    python_requires='>=3.7',
    install_requires=required_list,
    entry_points={
        "console_scripts": [
            "train_sc = sc.cmd.train_sc:main",
            "sc_generate_report = sc.report.generate_report:main",
        ]
    },
    scripts=[
        "sc/cmd/run_training.sh"
    ]
)

