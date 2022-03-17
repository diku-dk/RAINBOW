---
layout: page_documentation_single
title: "Ru"
filename: "matrix3.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/matrix3/ru
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
Creates a euler rotation matrix given oritation (radians) and 
the axis.   

## Example
```python
    radians = 0.5 * np.pi
    axis     = np.array([1,0,0]) 
    print(mat3.Ru(radians, axis))
```
output:
```bash
   [[ 1.000000e+00  0.000000e+00  0.000000e+00]
    [ 0.000000e+00  6.123234e-17 -1.000000e+00]
    [ 0.000000e+00  1.000000e+00  6.123234e-17]]
```