---
layout: page_documentation_single
title: "orthonomalize"
filename: "matrix3.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/matrix3/orthonomalize
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
Finds the orthogonal matrix of a given matrix M.  


## Example
```python
    M    = np.random.rand(3,3)
    M_orthonomalized = mat3.ortonormalize(M)
    expected = mat3.identity()
    print(np.dot(np.transpose(M_orthonomalized), M_orthonomalized))
```
output:
```bash
   [[1.00000000e+00 0.00000000e+00 5.55111512e-17]
    [0.00000000e+00 1.00000000e+00 1.66533454e-16]
    [5.55111512e-17 1.66533454e-16 1.00000000e+00]]
```