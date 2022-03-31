---
layout: page_documentation_single
title: "PCA Guide"
filename: "functions.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/functions/pca_guide
math: http://latex.codecogs.com/svg.latex? 
---
## Principal Component Analysis
This tutorial shows how to use our module's principal component analysis [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis). This implementation finds the eigenvalues using the NumPy LinAlg [eig](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html) library. 

## Examples 
First import the module
``` 
    import isl.math.functions as func
```
### Example 1
Then define a dataset having the signature [[x0,y0,z0], ... , [xN,yN,zN]]
```
    import numpy as np
    M = 10
    P = np.random.rand(M, 3)
    mean, principal_componets, eigenvectors = func.PCA(P)
    mean, principal_componets, eigenvectors
```
Output
```
    (array([0.36885196, 0.54575938, 0.46111808]),
     array([0.04772988, 0.11529473, 0.102832  ]),
     array([[ 0.8278388 ,  0.47242563, -0.30248461],
        [ 0.55871669, -0.74262082,  0.36925598],
        [ 0.05018538,  0.47468763,  0.87872241]]))
```
### Example 2
Find the most describing eigenvector
```
    import numpy as np
    M = 10
    P = np.random.rand(M, 3)
    eigenvector = func.direction_of_most_variance(P)
    eigenvector
```
Output
```
    array([-0.65208111, -0.62503103,  0.42909955])    
```        