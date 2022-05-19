---
layout: page_documentation_single
title: "inverse"
filename: "coordsys.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/coordsys/inverse
---
## Purpose & Params
The purpose of this function is to find the inverse coordinate transformation. This 
function is the inverse of <b>cs.xform_point</b>


## Example
```python
   x,y,z      = 1,0,5
   radians    = np.pi * 0.75
   axis       = np.array([0,-0.5,1])
   r          = vec3.make(x,y,z)
   q          = quat.Ru(radians, axis)
   coordsys   = cs.make(r,q)
   point      = np.array([1,0,0])
   point_rot  = cs.xform_point(coordsys, point)
   coordsys_inverse = cs.inverse(coordsys)        
   print(f"Point: {cs.xform_point(coordsys_inverse, point_rot)}" )
   print(f"Original point: {point}")
```
## Output
```bash
Point: [1. 0. 0.]
Original point: [1 0 0]
 ```   
   
   
   
   
   
   
   