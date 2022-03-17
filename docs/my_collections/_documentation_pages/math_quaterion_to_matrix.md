---
layout: page_documentation_single
title: "to_matrix"
filename: "quaternion.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/quaternion/to_matrix
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
Converts an euler rotation matrix to a quaternion.


## Example
```python
    import numpy as np
    # Cosider the case where we want to convert a quaternion to a euler 
    # rotation matrix, that rotates 90 degrees on the x-axis
    radian    = (0.5 * np.pi)/2
    q         = np.array([np.cos(radian), np.sin(radian), 0, 0])
    print(quat.to_matrix(q))
```           
output:       
```bash       
   [[ 1.00000000e+00  0.00000000e+00  0.00000000e+00]
    [ 0.00000000e+00  2.22044605e-16 -1.00000000e+00]
    [ 0.00000000e+00  1.00000000e+00  2.22044605e-16]]
```