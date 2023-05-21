from setuptools import setup
from Cython.Build import cythonize

#Simple setup script if one wishes to compile the pyton code into cython code.
#Run with python3 cython_setup.py build_ext --inplace

setup(
    ext_modules = cythonize("types.py", "api.py", "visualizer.py")
)
