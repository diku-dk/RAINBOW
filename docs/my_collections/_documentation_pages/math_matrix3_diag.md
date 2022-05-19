---
layout: page_documentation_single
title: "diag"
filename: "matrix3.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/matrix3/diag
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
Converts an 3 real values to a 3 by 3 diagonal matrix.  


## Example
```python
    a, b, c = 1,2,3
    np.diag(np.array([a,b,c]))
```
output:
```bash
   array([[1, 0, 0],
          [0, 2, 0],
          [0, 0, 3]])
```