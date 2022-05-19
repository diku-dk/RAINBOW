---
layout: page
title: "Installation Anaconda"
os: "Anaconda"
permalink: /installation_guide/anaconda/
---
## Install Anaconda
TBA
## Jupyter notebook setup
You need to setup your conda environment for the Jupyter notebooks
This is how to install and setup your conda environment

```bash
conda create -n libisl python=3.9
conda activate libisl
conda config --add channels conda-forge
conda install igl
conda install pyhull
conda install wildmeshing
conda install ipympl
conda install jupyter
```

When you want to run the jupyter notebook you have to do

```bash
conda activate libisl
jupyter notebook
```

### Python versions
The python version used by pybind11 to wrap our C++ code must match with the python version used in the Jupyter notebook otherwise one will get error messages like

```bash
ModuleNotFoundError: No module named 'pyisl'
```
The advice is to make sure the python version found in CMake is the same as the one used in the Jupyter notebooks.

### Latest version of wildmeshing 

Using conda to install wildmeshing is easy and convenient and will work for you if you do not use sizing fields, as of version 0.3.0.2. To get around this one should clone the latest version of master from

https://github.com/wildmeshing/wildmeshing-python

Then run CMake and build the wildmeshing pacakge. On my macboo the output file is named

```bash
wildmeshing.cpython-39-darwin.so
```

The name may vary depending on your OS and python version. Next copy the file from your build output into the "python" folder of libisl

```bash
cp wildmeshing.cpython-39-darwin.so libisl/python
```

Do not worry about wildmeshing installed with conda. This "local" version will