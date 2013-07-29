"""ftz: flush denormal numbers in numpy arrays to zero

Denormal numbers ("denorms") are extremely nonzero small numbers (less than
1.2-38 for single precision or 2.2e-308 in double precision) which are
handled poorly on most modern archicectures. Instructions involving denormal 
operands may run as much as 100 times slower than those involving standard
floating point operands [1], giving rise to very odd performance bugs.

This package contains optimized SSE code to flush denorms in numpy arrays
to zero.

[1] http://en.wikipedia.org/wiki/Denormal_number
"""

DOCLINES = __doc__.split("\n")

from distutils.core import setup
from distutils.extension import Extension
import numpy as np

CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD
Programming Language :: C
Programming Language :: Python
Operating System :: OS Independent
"""

try:
    from Cython.Distutils import build_ext
    src = ['ftz.pyx', 'src/ftz.c']
except ImportError:
    from distutils.command import build_ext
    src = ['ftz.c', 'src/ftz.c']

ext = Extension("ftz", src,
    include_dirs=[np.get_include(), 'include'],
    #removing O0 will lead to incorrect results
    extra_compile_args=['-msse', '-O0'])

setup(name='ftz',
  author='Robert McGibbon',
  author_email='rmcgibbo@gmail.com',
  url = "http://github.com/rmcgibbo/ftz",
  description=DOCLINES[0],
  long_description="\n".join(DOCLINES[2:]),
  version='0.1',
  license='BSD',
  cmdclass={'build_ext': build_ext},
  include_dirs = [np.get_include()],
  ext_modules = [ext]
)
