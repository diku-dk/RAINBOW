---
layout: page_documentation_single
title: "from_matrix"
filename: "quaternion.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/quaternion/from_matrix
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
Converts an euler rotation matrix to a quaternion.


## Example
```python
    import numpy as np
    # Cosider the case where we want to convert a euler roation matrix, that
    # rotates 90 degrees on the x-axis
    radian    = (0.5*np.pi)/2
    M  = np.array([
        [1,                               0,                                 0],
        [0,       (1-2*(np.sin(radian)**2)),  -2*np.cos(radian)*np.sin(radian)],
        [0, 2*np.cos(radian)*np.sin(radian), 1-2*(np.sin(radian)**2)          ]
    ])
    print(quat.from_matrix(M))
```           
output:       
```bash       
   array([    0.70710678, 0.70710678, 0.        , 0.        ])
```