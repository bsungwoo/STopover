from setuptools import setup, find_packages

setup(
    name="STopover",
    version="1.4.4",
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
)