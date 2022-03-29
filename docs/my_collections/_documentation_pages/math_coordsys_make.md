---
layout: page_documentation_single
title: "make"
filename: "coordsys.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/coordsys/make
---
## Purpose & Params
The purpose of the <b>make</b> function is to initiate a coordinate transformation. As input, the function takes an origin vector. For rotation, the function takes a quaternion.  

## Example
```python
    x,y,z      = 1,2,3                    # Define origin coordinate 
    radians    = np.pi * 0.5              # Define the orientation
    axis       = np.array([1,1,0])        # Define the axis of rotation
    r          = vec3.make(x,y,z)         # Define the origin vector
    q          = quat.Ru(radians, axis)   # Define the quaternion
    coordsys   = cs.make(r,q)             # make the coordinate transformtation
```