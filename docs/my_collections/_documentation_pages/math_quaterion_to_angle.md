---
layout: page_documentation_single
title: "to_angle"
filename: "quaternion.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/quaternion/to_angle
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
The purpose of to_angle is to find the angle of a given quaterion.


## Example
```python
    radian    = (0.5 * np.pi)/2
    axis      = np.array([0,1,0])
    axis_unit = axis / np.linalg.norm(axis)
    q  = quat.Ru(radian, axis_unit)
    print(quat.to_angle(q, axis_unit))
```           
output:       
```bash       
   0.7853981633974484
```