from setuptools import setup
from Cython.Build import cythonize

#Simple setup script if one wishes to compile the pyton code into cython code.
#Run with python3 cython_setup.py build_ext --inplace

setup(
    ext_modules = cythonize("quaternion.py", "angle.py", "vector3.py", "matrix3.py")
)
