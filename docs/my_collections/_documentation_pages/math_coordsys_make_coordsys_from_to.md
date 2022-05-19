---
layout: page_documentation_single
title: "make_coordsys_from_to"
filename: "coordsys.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/coordsys/make_coordsys_from_to
---
## Purpose & Params
Assumes that 'A' maps from bf_1 to wcs, and 'B' maps from bf_2 to wcs. Now compute the transform that maps from bf_1 to bf_2 

## Example
```python
    radians_A   = np.pi * 0.5
    radians_B   = np.pi * 0.75
    radians_W   = np.pi * 0.33
    axis_A      = np.array([0,0,1])
    axis_B      = np.array([1,0,0])
    axis_W      = np.array([0,1,0])
    r_A         = vec3.make(0,1,0)
    r_B         = vec3.make(1,0,1)
    r_W         = vec3.make(0,1,1)
    q_A         = quat.Ru(radians_A, axis_A)
    q_B         = quat.Ru(radians_B, axis_B)
    q_W         = quat.Ru(radians_W, axis_W)
    bf_1        = cs.make(r_A, q_A)
    bf_2        = cs.make(r_B, q_B)
    wcs         = cs.make(r_W, q_W)        
    A           = cs.make_coordsys_from_to(bf_1, wcs)
    B           = cs.make_coordsys_from_to(bf_2, wcs)
    point       = vec3.make(1.5,-700,0.001)
    coord_A2B   = cs.make_coordsys_from_to(A, B)
    coord_B2A   = cs.make_coordsys_from_to(B, A)
    print(f"Point: {cs.xform_point(coord_B2A, cs.xform_point(coord_A2B, point))}")
    print(f"Original Point: {point}")
```


```bash
Point: [ 1.5e+00 -7.0e+02  1.0e-03]
Original Point: [ 1.5e+00 -7.0e+02  1.0e-03]
```