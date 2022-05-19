---
layout: page_documentation_single
title: "diag_from_array"
filename: "matrix3.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/matrix3/diag_from_array
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
Converts an array of length 3 to a 3 by 3 diagonal matrix.  


## Example
```python
    v = np.array([1,2,3])
    print(mat3.diag_from_array(v)
```
output:
```bash
   [[1. 0. 0.]
    [0. 2. 0.]
    [0. 0. 3.]] 
```