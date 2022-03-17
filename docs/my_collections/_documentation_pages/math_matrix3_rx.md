---
layout: page_documentation_single
title: "Rx"
filename: "matrix3.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/matrix3/rx
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
Rotate a coordinate on the x-axis given a euler rotation matrix.  

## Example
```python
    # Rotates the coordinate v 90 degrees on the x-axis
    v = np.array([0,0,1])
    radian_x        = 0.5 * np.pi
    rotation_matrix = mat3.Rx(radian_x)
    print(np.dot(rotation_matrix, v))
```
output:
```bash
   [ 0.000000e+00 -1.000000e+00  6.123234e-17]
```