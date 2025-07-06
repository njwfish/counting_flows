from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the Cython extension
extensions = [
    Extension(
        "sampling.distribute_shift_sparse",  # module name
        ["sampling/distribute_shift_sparse.pyx"],  # source file
        include_dirs=[np.get_include()],  # include NumPy headers
        language="c++",
        extra_compile_args=["-O3", "-ffast-math"],  # optimization flags
    )
]

setup(
    name="counting_flows_cython",
    ext_modules=cythonize(extensions, compiler_directives={
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
        'language_level': 3
    }),
    zip_safe=False,
) 