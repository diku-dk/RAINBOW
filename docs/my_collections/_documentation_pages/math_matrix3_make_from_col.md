---
layout: page_documentation_single
title: "make_from_cols"
filename: "matrix3.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/matrix3/make_from_cols
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
Converts an 3 numpy arrays of length 3 each to a 3 by 3 matrix. Every array
are used as columns.  


## Example
```python
    a,d,h = (np.array([1,4,7]),
         np.array([2,5,8]),
         np.array([3,6,9])
                 )        
    print(mat3.make_from_cols(a,d,h))
```
output:
```bash
   array([[1., 2., 3.],
          [4., 5., 6.],
          [7., 8., 9.]])
```