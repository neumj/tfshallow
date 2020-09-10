from setuptools import setup, find_packages

reqs = [
    "h5py",
    "jupyterlab",
    "matplotlib",
    "numpy",
    "pandas",
    "pillow",
    "scipy==1.2.1",
    "scikit-learn",
    "yaml",
    "tensorflow==1.15",
    "imageio"
]

conda_reqs = [
    "h5py",
    "jupyterlab",
    "matplotlib",
    "numpy",
    "pandas",
    "pillow",
    "scipy==1.2.1",
    "scikit-learn",
    "yaml",
    "tensorflow==1.15",
    "imageio"
]

test_pkgs = []

setup(
    name="tfshallow",
    python_requires='>3.4',
    description="Package for tensor flow neural network experimentation",
    url="https://github.com/neumj/neural-network",
    install_requires=reqs,
    conda_install_requires=conda_reqs,
    test_requires=test_pkgs,
    packages=find_packages(),
    include_package_data=True
)
