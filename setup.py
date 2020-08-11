from setuptools import setup, find_packages

with open("requirements.txt") as f_req:
    required_list = [line.rstrip() for line in f_req.readlines()]

setup(
    name='sc',
    version='0.1',
    package_dir={"": "sc"},
    packages=find_packages(where="sc"),
    url='https://www.bnl.gov/cfn/',
    license='GPL',
    author='Xiaohui Qu, Matthew Carbone',
    author_email='xiaqu@bnl.gov; mrc2215@columbia.edu',
    description='Semi-supervised Clustering',
    python_requires='>=3.7',
    install_requires=required_list
)
