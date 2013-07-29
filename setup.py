import os
import sys
import tempfile
import shutil

from distutils.ccompiler import new_compiler
from distutils.core import setup
from distutils.extension import Extension

import numpy as np
from Cython.Distutils import build_ext

ext = Extension("ftz", ["ftz.pyx", "src/ftz.c"],
    include_dirs=[np.get_include(), 'include'],
    #removing O0 will lead to incorrect results
    extra_compile_args=['-msse', '-O0'])

setup(name='Flush denorms to zero',
  cmdclass={'build_ext': build_ext},
  include_dirs = [np.get_include()],  
  ext_modules = [ext]
)
