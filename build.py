import os
import shutil
from setuptools import Distribution, Extension
import numpy as np  # Ensure you have NumPy imported
from Cython.Build import build_ext, cythonize

cython_dir = os.path.join("sp")
extension = Extension(
    "sp.sp",
    [
        os.path.join(cython_dir, "sp_funcs.pyx"),
    ],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3"],
)

ext_modules = cythonize([extension], include_path=[cython_dir])
dist = Distribution({"ext_modules": ext_modules})
cmd = build_ext(dist)
cmd.ensure_finalized()
cmd.run()

for output in cmd.get_outputs():
    relative_extension = os.path.relpath(output, cmd.build_lib)
    shutil.copyfile(output, relative_extension)
