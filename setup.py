import os
import sys
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

# Do not import pybind11 here; it will be imported inside the build extension class.

# Define the extension module with all necessary source files
ext_modules = [
    Extension(
        "spatial_analysis",  # Name of the generated Python module
        sources=[
            "src/type_conversion.cpp",
            "src/topological_comp.cpp",
            "src/jaccard.cpp",
            "src/parallelize.cpp"  # Main file including all parallel functions
        ],
        include_dirs=[
            # Include directories will be set in the custom build extension class
        ],
        language="c++",
        extra_compile_args=["-O3", "-Wall", "-std=c++17", "-fopenmp"],  # Optimization and OpenMP for parallelism
        extra_link_args=["-fopenmp"]  # Ensure OpenMP is linked
    ),
]

# Define a custom build extension to handle compiler specifics and pybind11
class BuildExt(build_ext):
    def build_extensions(self):
        # Import pybind11 here, after build dependencies have been installed
        import pybind11

        # Add pybind11 include directories
        for ext in self.extensions:
            ext.include_dirs.extend([
                pybind11.get_include(),
                pybind11.get_include(user=True),
                os.path.join(sys.prefix, "include", "eigen3")  # Path to Eigen library
            ])

        # Apply compiler-specific options for GCC or Clang on Unix-based systems
        compiler = self.compiler.compiler_type
        if compiler == "unix":
            for ext in self.extensions:
                ext.extra_compile_args += ["-D_GLIBCXX_USE_CXX11_ABI=0"]
        elif compiler == "msvc":
            for ext in self.extensions:
                ext.extra_compile_args = ["/O2", "/openmp"]
        super().build_extensions()

# Final setup function including Pybind11 extension and existing Python package configuration
setup(
    name="STopover",
    version="1.5.0",
    description="Tool to capture spatial colocalization and interaction in the TME using topological analysis in spatial transcriptomics data",
    url="https://github.com/bsungwoo/STopover.git",
    author="Sungwoo Bae, Hyekyoung Lee, Hongyoon Choi",
    packages=find_packages(include=['STopover', 'STopover.*']),
    include_package_data=True,
    package_data={'': ['data/*.txt', 'data/*.csv', 'app/image/*']},
    install_requires=[
        "scanpy~=1.9",
        "pyqt5~=5.15.7",
        "leidenalg",
        "pyarrow",
        "ply",
        "pytest",
        "parmap~=1.6"
    ],
    ext_modules=ext_modules,  # Include the C++ extension
    cmdclass={"build_ext": BuildExt},  # Use custom build_ext
    zip_safe=False,
)