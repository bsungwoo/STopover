# STopover
Tool to capture spatial colocalization and interaction in the TME using topological analysis in spatial transcriptomics data  
* Cite as: Bae S, Lee H, Na KJ, Lee DS, Choi H, Kim YT. STopover captures spatial colocalization and interaction in the tumor microenvironment using topological analysis in spatial transcriptomics data. bioRxiv, 2022.  
* https://doi.org/10.1101/2022.11.16.516708  

## Installation and running
### 1. Python
#### Install conda environment and add jupyter kernel
```Plain Text  
  conda create -n STopover python=3.8
  conda activate STopover
  pip install git+https://github.com/bsungwoo/STopover.git
  pip install jupyter
  # Install compatible version of ipywidgets with parmap (~=1.6.0)
  pip install ipywidgets~=7.7.0
  python -m ipykernel install --user --name STopover --display-name STopover
```
#### Run GUI for STopover (PyQt)
```Plain Text
  conda activate STopover
  python
  from STopover import app
  app.main()
```
### 2. R
```Plain Text
  # Install BiocManager dependencies
  if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
  BiocManager::install("clusterProfiler")
  BiocManager::install("org.Mm.eg.db")
  BiocManager::install("org.Hs.eg.db")
  
  # Install STopover
  devtools::install_github("bsungwoo/STopover", build_vignettes = T, force = T)
```
### 3. Standalone app (packaged with [pyinstaller](https://github.com/pyinstaller/pyinstaller))  
Please download file: [STopover_v1_windows.exe]()  

## Key packages
** Python  
Please refer to yaml file: [Python requirements](https://github.com/bsungwoo/STopover/blob/master/STopover_env.yaml)  
** R  
Please refer to DESCRIPTION file: [R requirements](https://github.com/bsungwoo/STopover/blob/master/DESCRIPTION)  

## Usage
### Python Code Example
Please refer to [README_Python.md](https://github.com/bsungwoo/STopover/blob/master/STopover/README_Python.md)  
### R Code Example
Please refer to vignettes in the package and the below documents  
```Plain Text
  library(STopover)
  ??STopover  # Explanation for topological similarity and short examples
  browseVignettes("STopover")  # Browse for the vignettes (Rmd files in vignettes)
```
Link to documents: [STopover_visium](https://rpubs.com/bsungwoo/STopover_visium), [STopover_cosmx](https://rpubs.com/bsungwoo/STopover_cosmx)  
