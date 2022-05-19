---
layout: page_documentation_single
title: "Rz"
filename: "matrix3.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/matrix3/rz
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
Rotate a coordinate on the z-axis given a euler rotation matrix.  

## Example
```python
    # Rotates the coordinate v 90 degrees on the z-axis
    v = np.array([1,0,0])
    radian_z        = 0.5 * np.pi
    rotation_matrix = mat3.Rz(radian_z)
    print(np.dot(rotation_matrix, v))
```
output:
```bash
   [6.123234e-17 1.000000e+00 0.000000e+00]
```