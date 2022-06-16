# STopover: graph filtration for extraction of spatial overlap patterns in spatial transcriptomic data

## Optimal parameter choices  
  minimum size of connected components (min_size) = 20  
  Full width half maximum of Gaussian smoothing kernel (fwhm) = 2.5  
  Lower percentile value threshold to remove the connected components (thres_per) = 30  

## Code Example  
```Plain Text

```

## Python for STopover implementation    
### Install conda environment and add jupyter kernel  
```Plain Text  
  conda create --n STopover -c conda-forge graph-tool=2.45 python=3.7  
  conda activate STopover  
  pip install git+https://github.com/bsungwoo/STopover.git  
  python -m ipykernel install --user --name STopover --display-name STopover  
```

### Dependency (python)  
```Plain Text
python 3.7  
scanpy 1.5.1  
numpy 1.21.2  
pandas 1.3.2  
h5py 2.10.0  
scikit-learn 0.24.2  
scipy 1.7.1  
graph-tool 2.45  
```