from setuptools import setup, find_packages

setup(
    name = "STopover",
    version = "1.0.1",
    description = "Graph filtration for extraction of spatial overlap patterns in spatial transcriptomic data",
    url = "https://github.com/bsungwoo/STopover.git",
    author = "Sungwoo Bae, Hyekyoung Lee, Hongyoon Choi",
    packages=find_packages(include=['STopover', 'STopover.*']),
    install_requires = ["scanpy==1.5.1","pandas==1.3.2","numpy==1.21.2",
                        "h5py==2.10.0", "scikit-learn==0.24.2",
                        "scipy==1.7.1", "jupyter"]
)