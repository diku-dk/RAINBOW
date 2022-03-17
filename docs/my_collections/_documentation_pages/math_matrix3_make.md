---
layout: page_documentation_single
title: "make"
filename: "matrix3.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/matrix3/make
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
Converts an 9 real values to a 3 by 3 matrix.  


## Example
```python
    a,b,c,d,e,f,h,i,j = 1,2,3,4,5,6,7,8,9
    mat3.make(a,b,c,d,e,f,h,i,j)
```
output:
```bash
   array([[1., 2., 3.],
          [4., 5., 6.],
          [7., 8., 9.]])
```