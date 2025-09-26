"""
Install the cython first as follows:

python3 install_me_first.py build_ext --inplace    

"""
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

ext_modules = [Extension("evaluations", ["evaluations.pyx"])]
setup(ext_modules=cythonize(ext_modules))
# setup(ext_modules=cythonize('evaluations.pyx'))