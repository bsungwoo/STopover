import os
import sys
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

# Attempt to import pybind11 and install if not found
try:
    import pybind11
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11"])
    import pybind11  # Re-import after installation

# Explicit path to the Eigen include directory
EIGEN_INCLUDE_DIR = "/opt/conda/envs/STopover_test/include/eigen3"

# Function to install Eigen with conda if it's not already in the specified directory
def install_eigen_with_conda():
    if not os.path.exists(os.path.join(EIGEN_INCLUDE_DIR, "Eigen", "Core")):
        print("Eigen library not found. Attempting to install with Conda...")
        try:
            subprocess.check_call(["conda", "install", "-y", "-c", "conda-forge", "eigen"])
        except subprocess.CalledProcessError:
            raise RuntimeError("Conda installation of Eigen failed. "
                               "Ensure conda is installed or install Eigen manually.")

# Ensure Eigen is installed before proceeding
install_eigen_with_conda()

# Define the extension module with all necessary source files
ext_modules = [
    Extension(
        "STopover.parallelize",  # Module name within the STopover package
        sources=[
            "src/type_conversion.cpp",
            "src/topological_comp.cpp",
            "src/jaccard.cpp",
            "src/parallelize.cpp",
            "src/make_dendrogram_bar.cpp",
            "src/make_original_dendrogram.cpp",
            "src/make_smoothed_dendrogram.cpp"
        ],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            EIGEN_INCLUDE_DIR,  # Explicit Eigen directory
            "src"  # Assuming headers are in 'src'
        ],
        language="c++",
        extra_compile_args=["-O3", "-Wall", "-std=c++17", "-fopenmp"],
        extra_link_args=["-fopenmp"]
    ),
]

# Define a custom build extension to handle compiler specifics
class BuildExt(build_ext):
    def build_extensions(self):
        # Apply compiler-specific options for GCC or Clang on Unix-based systems
        compiler = self.compiler.compiler_type
        if compiler == "unix":
            for ext in self.extensions:
                # Ensures ABI compatibility if needed
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