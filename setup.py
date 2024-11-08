import os
import sys
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

# Attempt to import pybind11 and install if not found
try:
    import pybind11
except ImportError:
    print("Pybind11 not found. Installing pybind11...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11"])
    import pybind11  # Re-import after installation

def install_eigen_with_conda(conda_prefix):
    """
    Install Eigen using Conda if it's not already installed.
    """
    eigen_include = os.path.join(conda_prefix, 'include', 'eigen3')
    if not os.path.exists(os.path.join(eigen_include, "Eigen", "Core")):
        print("Eigen library not found. Attempting to install with Conda...")
        try:
            subprocess.check_call(["conda", "install", "-y", "-c", "conda-forge", "eigen"])
        except subprocess.CalledProcessError:
            raise RuntimeError("Conda installation of Eigen failed. "
                               "Ensure conda is installed or install Eigen manually.")
        
        # Re-verify after installation
        if not os.path.exists(os.path.join(eigen_include, "Eigen", "Core")):
            raise FileNotFoundError(
                f"Eigen library not found in expected directory after installation: {eigen_include}\n"
                f"Please ensure Eigen is installed in the directory."
            )
    return eigen_include

def find_eigen_include():
    """
    Dynamically find the Eigen include directory within the active Conda environment.
    Allows user override via EIGEN_INCLUDE environment variable.
    """
    # Allow user to specify Eigen include path via environment variable
    user_eigen = os.environ.get('EIGEN_INCLUDE')
    if user_eigen and os.path.exists(os.path.join(user_eigen, "Eigen", "Core")):
        print(f"Using Eigen include directory from EIGEN_INCLUDE environment variable: {user_eigen}")
        return user_eigen
    
    # Attempt to use CONDA_PREFIX
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        print(f"Detected CONDA_PREFIX: {conda_prefix}")
        eigen_include = os.path.join(conda_prefix, 'include', 'eigen3')
    else:
        # Fallback to sys.prefix
        print(f"CONDA_PREFIX not set. Using sys.prefix: {sys.prefix}")
        eigen_include = os.path.join(sys.prefix, 'include', 'eigen3')
    
    # Verify that the Eigen directory exists
    if not os.path.exists(os.path.join(eigen_include, "Eigen", "Core")):
        # Attempt to install Eigen
        eigen_include = install_eigen_with_conda(conda_prefix if conda_prefix else sys.prefix)
    
    print(f"Using Eigen include directory: {eigen_include}")
    return eigen_include

# Dynamically find Eigen include directory
EIGEN_INCLUDE_DIR = find_eigen_include()

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
            EIGEN_INCLUDE_DIR,  # Correct Eigen directory
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