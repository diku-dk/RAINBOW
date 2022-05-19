---
layout: page_documentation_single
title: "Ry"
filename: "matrix3.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/matrix3/ry
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
Rotate a coordinate on the y-axis given a euler rotation matrix.  

## Example
```python
    # Rotates the coordinate v 90 degrees on the y-axis
    v = np.array([1,0,0])
    radian_y        = 0.5 * np.pi
    rotation_matrix = mat3.Ry(radian_y)
    print(np.dot(rotation_matrix, v))
```
output:
```bash
   [ 6.123234e-17  0.000000e+00 -1.000000e+00]
```