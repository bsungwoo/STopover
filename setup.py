from setuptools import setup, find_packages

setup(
    name = "STopover",
    version = "1.0.4",
    description = "Graph filtration for extraction of spatial overlap patterns in spatial transcriptomic data",
    url = "https://github.com/bsungwoo/STopover.git",
    author = "Sungwoo Bae, Hyekyoung Lee, Hongyoon Choi",
    packages = find_packages(include=['STopover', 'STopover.*']),
    include_package_data = True,
    package_data = {'': ['data/*.txt']},
    install_requires = ["scanpy~=1.9.0","pandas~=1.4.3","numpy~=1.20.3","matplotlib~=3.4.3",
                        "scikit-learn~=0.24.2","scipy~=1.7.3","networkx~=2.6.3","pyqt5~=5.15.7",
                        "pyarrow", "jupyter", "ply", "pytest"]
)