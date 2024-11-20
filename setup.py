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
    
    Args:
        conda_prefix (str): The prefix path of the active Conda environment.
        
    Returns:
        str: The path to the Eigen include directory.
        
    Raises:
        RuntimeError: If Conda installation fails.
        FileNotFoundError: If Eigen is not found after installation.
    """
    print("Attempting to install Eigen via Conda...")
    try:
        subprocess.check_call(["conda", "install", "-y", "-c", "conda-forge", "eigen"])
    except subprocess.CalledProcessError:
        raise RuntimeError("Conda installation of Eigen failed. "
                           "Ensure Conda is installed and accessible, or install Eigen manually.")

    eigen_include = os.path.join(conda_prefix, 'include', 'eigen3')
    if not os.path.exists(os.path.join(eigen_include, "Eigen", "Core")):
        raise FileNotFoundError(
            f"Eigen library not found in expected directory after installation: {eigen_include}\n"
            f"Please ensure Eigen is installed in the directory."
        )
    print(f"Eigen successfully installed in: {eigen_include}")
    return eigen_include

def find_eigen_include():
    """
    Dynamically find the Eigen include directory within the active Conda environment or system paths.
    Allows user override via EIGEN_INCLUDE environment variable.
    
    Returns:
        str: The path to the Eigen include directory.
        
    Raises:
        FileNotFoundError: If Eigen is not found in any of the expected directories.
    """
    # 1. Allow user to specify Eigen include path via environment variable
    user_eigen = os.environ.get('EIGEN_INCLUDE')
    if user_eigen:
        eigen_include = user_eigen
        if os.path.exists(os.path.join(eigen_include, "Eigen", "Core")):
            print(f"Using Eigen include directory from EIGEN_INCLUDE: {eigen_include}")
            return eigen_include
        else:
            raise FileNotFoundError(
                f"Eigen library not found in the directory specified by EIGEN_INCLUDE: {eigen_include}\n"
                f"Please ensure the path is correct or unset EIGEN_INCLUDE to auto-detect."
            )

    # 2. Check Conda environment
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        eigen_include = os.path.join(conda_prefix, 'include', 'eigen3')
        if os.path.exists(os.path.join(eigen_include, "Eigen", "Core")):
            print(f"Found Eigen include directory in Conda environment: {eigen_include}")
            return eigen_include
        else:
            # Attempt to install Eigen via Conda
            eigen_include = install_eigen_with_conda(conda_prefix)
            return eigen_include

    # 3. Check sys.prefix (useful for virtualenv or non-Conda environments)
    eigen_include = os.path.join(sys.prefix, 'include', 'eigen3')
    if os.path.exists(os.path.join(eigen_include, "Eigen", "Core")):
        print(f"Found Eigen include directory in sys.prefix: {eigen_include}")
        return eigen_include

    # 4. Check standard system paths
    standard_paths = [
        '/usr/include/eigen3',
        '/usr/local/include/eigen3',
        '/opt/include/eigen3',
    ]
    for path in standard_paths:
        if os.path.exists(os.path.join(path, "Eigen", "Core")):
            print(f"Found Eigen include directory in standard path: {path}")
            return path

    # 5. Attempt to install Eigen via Conda if possible
    if conda_prefix:
        eigen_include = install_eigen_with_conda(conda_prefix)
        return eigen_include

    # 6. If Eigen still not found, raise error
    raise FileNotFoundError(
        "Eigen library not found in any of the expected directories.\n"
        "Please install Eigen via Conda:\n"
        "    conda install -c conda-forge eigen\n"
        "Or specify the Eigen include directory via the EIGEN_INCLUDE environment variable."
    )

# Dynamically find Eigen include directory
try:
    EIGEN_INCLUDE_DIR = find_eigen_include()
except FileNotFoundError as e:
    print(str(e))
    sys.exit(1)

# Define the extension module with all necessary source files
ext_modules = [
    Extension(
        "STopover.parallelize",  # Module name within the STopover package
        sources=[
            "src/topological_comp.cpp",
            "src/jaccard.cpp",
            "src/parallelize.cpp",
            "src/make_dendrogram_bar.cpp",
            "src/make_original_dendrogram.cpp",
            "src/make_smoothed_dendrogram.cpp",
            "src/utils.cpp",
            # "src/ThreadPool.cpp",
            # "src/logger.cpp",
            # "src/custom_streambuf.cpp",
            # "src/cout_redirector.cpp"
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
    Extension(
        "STopover.connected_components",  # Another module name
        sources=[
            "src/make_original_dendrogram.cpp",
            "src/make_smoothed_dendrogram.cpp",
            "src/make_dendrogram_bar.cpp",
            "src/topological_comp.cpp",
            "src/connected_components.cpp"
        ],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            EIGEN_INCLUDE_DIR,
            "src"
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