---
layout: page_documentation_single
title: "make_from_rows"
filename: "matrix3.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/matrix3/make_from_rows
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
Converts an 3 numpy arrays of length 3 each to a 3 by 3 matrix. Every array
are used as rows.  


## Example
```python
    a,d,h = (np.array([1,2,3]),
             np.array([4,5,6]),
             np.array([7,8,9])
                 )     
mat3.make_from_rows(a,d,h)
```
output:
```bash
   array([[1., 2., 3.],
          [4., 5., 6.],
          [7., 8., 9.]])
```