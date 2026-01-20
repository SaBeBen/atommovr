from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import os
import sys

# Path to output directory
output_dir = os.path.abspath("")
os.makedirs(output_dir, exist_ok=True)

# Define the extension as a shared library (not a Python importable module)
# Path to output directory
output_dir = os.path.abspath("")
os.makedirs(output_dir, exist_ok=True)

extra_compile_args = []
if sys.platform == "win32":
    # MSVC specific flags (example: disable warning C4100: unreferenced formal parameter)
    extra_compile_args=['/wd4100', '/wd4101']
else:
    # GCC/Clang flags
    extra_compile_args = [
        "-fPIC",
        "-Wno-unused-but-set-variable",
        "-Wno-unused-variable",
        "-Wno-sign-compare",
        "-Wno-unreachable-code",
    ]

# Define the extension as a shared library (not a Python importable module)
c_extension = Extension(
    name="libmatching_placeholder",  # name won't be used for output
    sources=[
        "bottleneckBipartiteMatching.c",
        "matrixUtils.c",
        "mmio.c",
        "extern/cheap.c",
        "extern/matching.c",
    ],
    include_dirs=["extern"],
    extra_compile_args=extra_compile_args,
    libraries=[] if sys.platform == "win32" else ["m"],
)

class BuildSharedLibrary(build_ext):
    def build_extensions(self):
        # Compile source files into object files
        for ext in self.extensions:
            objects = self.compiler.compile(
                ext.sources,
                include_dirs=ext.include_dirs,
                extra_postargs=ext.extra_compile_args,
            )

            # Link into a shared object
            lib_path = os.path.join(output_dir, "libmatching_for_PPSU.so")
            self.compiler.link_shared_object(
                objects, lib_path, libraries=ext.libraries
            )
            print(f"Built shared library: {lib_path}")

setup(
    name="libmatching_for_PPSU",
    version="0.1",
    ext_modules=[c_extension],
    cmdclass={"build_ext": BuildSharedLibrary},
)
