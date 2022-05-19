---
layout: page_documentation_single
title: "rotate_array"
filename: "quaternion.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/quaternion/rotate_array
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
For information about rotation look at the documentation for rotation of 
quaterion, rotate.
The purpose of this function is to rotate multiple points with the same degree and around the same axis. 

## Example
```python
    # Rotate three coordinates 90 degrees around the y axis
    radian    = (0.5 * np.pi)/2
    axis      = np.array([0,1,0])
    axis_unit = axis / np.linalg.norm(axis)
    q  = quat.make(np.cos(radian),axis_unit[0] * np.sin(radian),axis_unit[1] * np.sin(radian),axis_unit[2] * np.sin(radian))
    rs = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ], dtype=np.float64)
    print(quat.rotate_array(q, rs))
```
output:
```bash
   [[ 2.22044605e-16  0.00000000e+00 -1.00000000e+00]
    [ 0.00000000e+00  1.00000000e+00  0.00000000e+00]
    [ 1.00000000e+00  0.00000000e+00  2.22044605e-16]] 
```