# STopover: capturing spatial colocalization and interaction in the TME using topological analysis in spatial transcriptomics data  

## Python for implementation  
### Install conda environment and add jupyter kernel  
```Plain Text  
  conda create -n STopover python=3.8
  conda activate STopover
  pip install git+https://github.com/bsungwoo/STopover.git@dev
  pip install jupyter
  python -m ipykernel install --user --name STopover --display-name STopover
```
### Run GUI for STopover (PyQt)  
```Plain Text
conda activate STopover
python
from STopover import app
app.main()
```
### Dependency (python)  
```Plain Text
python 3.8
scanpy 1.9.1
numpy 1.20.3
pandas 1.4.3
matplotlib 3.4.3
pyarrow 8.0.0
pyqt5 5.15.7
scipy 1.7.3
```
### Python Code Example  
Please refer to [README_Python.md](https://github.com/bsungwoo/STopover/blob/dev/STopover/README_Python.md)  

## R for implementation  
### Install R package and open vignettes  
```Plain Text  
  devtools::install_github("bsungwoo/STopover@dev", build_vignettes = T, force = T)  
  library(STopover)  
  ??STopover  # Explanation for topological similarity and short examples  
  browseVignettes("STopover")  # Browse for the vignettes (Rmd files in vignettes)
```
### Dependency (R)  
```Plain Text
Seurat 4.1.1
reticulate 1.25
dplyr 1.0.9
patchwork 1.1.1
```

## R Code Example  
Please refer to vignettes in the package  
