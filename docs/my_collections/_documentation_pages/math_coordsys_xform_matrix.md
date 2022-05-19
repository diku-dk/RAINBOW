---
layout: page_documentation_single
title: "xform_matrix"
filename: "coordsys.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/coordsys/xform_matrix
---
## Purpose & Params
The purpose of this function is to transform a matrix to another basis.  

## Example
```python
    x,y,z          = 1,0,0
    radians        = np.pi / 2
    axis           = np.array([0,0,1])
    r              = vec3.make(x,y,z)
    q              = quat.Ru(radians, axis)
    coordsys       = cs.make(r,q)
    identity_basis = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ])
    expected       = np.array([
        [0,-1,0],
        [1, 0,0],
        [0, 0,1]
    ])
    transformet_m = cs.xform_matrix(coordsys, identity_basis)
```
```bash
array([[ 0.0, -1.0,  0.0],
       [ 1.0,  0.0,  0.0],
       [ 0.0,  0.0,  1.0]])
```