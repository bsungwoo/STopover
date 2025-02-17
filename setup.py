import os
import sys
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

# Ensure pybind11 is available.
try:
    import pybind11
except ImportError:
    print("Pybind11 not found. Installing pybind11...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11"])
    import pybind11  # Re-import after installation

def install_eigen_with_conda(conda_prefix):
    print("Attempting to install Eigen via Conda...")
    try:
        subprocess.check_call(["conda", "install", "-y", "-c", "conda-forge", "eigen"])
    except subprocess.CalledProcessError:
        raise RuntimeError("Conda installation of Eigen failed. Ensure Conda is installed and accessible.")
    eigen_include = os.path.join(conda_prefix, 'include', 'eigen3')
    if not os.path.exists(os.path.join(eigen_include, "Eigen", "Core")):
        raise FileNotFoundError(
            f"Eigen library not found in expected directory: {eigen_include}"
        )
    print(f"Eigen successfully installed in: {eigen_include}")
    return eigen_include

def find_eigen_include():
    user_eigen = os.environ.get('EIGEN_INCLUDE')
    if user_eigen:
        if os.path.exists(os.path.join(user_eigen, "Eigen", "Core")):
            print(f"Using Eigen include directory from EIGEN_INCLUDE: {user_eigen}")
            return user_eigen
        else:
            raise FileNotFoundError(f"Eigen not found in EIGEN_INCLUDE: {user_eigen}")

    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        eigen_include = os.path.join(conda_prefix, 'include', 'eigen3')
        if os.path.exists(os.path.join(eigen_include, "Eigen", "Core")):
            print(f"Found Eigen in Conda environment: {eigen_include}")
            return eigen_include
        else:
            eigen_include = install_eigen_with_conda(conda_prefix)
            return eigen_include

    eigen_include = os.path.join(sys.prefix, 'include', 'eigen3')
    if os.path.exists(os.path.join(eigen_include, "Eigen", "Core")):
        print(f"Found Eigen in sys.prefix: {eigen_include}")
        return eigen_include

    for path in ['/usr/include/eigen3', '/usr/local/include/eigen3', '/opt/include/eigen3']:
        if os.path.exists(os.path.join(path, "Eigen", "Core")):
            print(f"Found Eigen in standard path: {path}")
            return path

    if conda_prefix:
        eigen_include = install_eigen_with_conda(conda_prefix)
        return eigen_include

    raise FileNotFoundError("Eigen library not found. Install via Conda or specify via EIGEN_INCLUDE.")

try:
    EIGEN_INCLUDE_DIR = find_eigen_include()
except FileNotFoundError as e:
    print(str(e))
    sys.exit(1)

ext_modules = [
    Extension(
        "STopover.parallelize",
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
            EIGEN_INCLUDE_DIR,
            "src"
        ],
        language="c++",
        extra_compile_args=["-O3", "-Wall", "-std=c++17", "-fopenmp"],
        extra_link_args=["-fopenmp"]
    ),
]

class BuildExt(build_ext):
    def build_extensions(self):
        compiler = self.compiler.compiler_type
        if compiler == "unix":
            for ext in self.extensions:
                # Optionally, remove the ABI flag if causing issues
                # ext.extra_compile_args += ["-D_GLIBCXX_USE_CXX11_ABI=0"]
                pass
        elif compiler == "msvc":
            for ext in self.extensions:
                ext.extra_compile_args = ["/O2", "/openmp"]
        super().build_extensions()

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
        "pybind11",
        "scanpy~=1.9",
        "pyqt5~=5.15.7",
        "leidenalg",
        "pyarrow",
        "ply",
        "pytest",
        "parmap~=1.6"
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)