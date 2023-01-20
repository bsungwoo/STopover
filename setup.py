from setuptools import setup, find_packages

setup(
    name = "STopover",
    version = "1.0.7",
    description = "Tool to capture spatial colocalization and interaction in the TME using topological analysis in spatial transcriptomics data",
    url = "https://github.com/bsungwoo/STopover.git@dev",
    author = "Sungwoo Bae, Hyekyoung Lee, Hongyoon Choi",
    packages = find_packages(include=['STopover', 'STopover.*']),
    include_package_data = True,
    package_data = {'': ['data/*.txt', 'app/image/*']},
    install_requires = ["scanpy~=1.9.0","pandas~=1.4.3","numpy~=1.20.3","matplotlib~=3.4.3",
                        "scipy~=1.7.3","pyqt5~=5.15.7","leidenalg",
                        "pyarrow", "ply", "pytest"]
)
