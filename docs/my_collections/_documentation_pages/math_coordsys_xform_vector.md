---
layout: page_documentation_single
title: "xfrom_vector"
filename: "coordsys.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/coordsys/xform_vector
---
## Purpose & Params
The purpose of this function is to transform a vector to another basis.  

## Example
```python
    x,y,z      = 0,-2,0
    radians    = np.pi / 2
    axis       = np.array([0,0,1])
    r          = vec3.make(x,y,z)
    q          = quat.Ru(radians, axis)
    coordsys   = cs.make(r,q)
    vec        = np.array([0,2,0])
    vec_rot    = cs.xform_vector(coordsys, vec)
```