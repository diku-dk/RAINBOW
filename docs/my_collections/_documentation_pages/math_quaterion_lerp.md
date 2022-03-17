---
layout: page_documentation_single
title: "lerp"
filename: "quaternion.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/quaternion/lerp 
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
The purpose of lerp is to find a linear interpolation of two quaterions. 


## Example
```python
    radian    = (0.5 * np.pi)/2
    axis_1      = np.array([0,1,0])
    axis_2      = np.array([1,0,0])
    axis_unit_1 = axis_1 / np.linalg.norm(axis_1)
    axis_unit_2 = axis_2 / np.linalg.norm(axis_2)
    q_1  = quat.make(np.cos(radian),axis_unit_1[0] * np.sin(radian),axis_unit_1[1] * np.sin(radian),axis_unit_1[2] * np.sin(radian))
    q_2  = quat.make(np.cos(radian),axis_unit_2[0] * np.sin(radian),axis_unit_2[1] * np.sin(radian),axis_unit_2[2] * np.sin(radian))
    h    = 0.5
    print(quat.lerp(q_1,q_2,h))
```           
output:       
```bash       
  [0.70710678 0.35355339 0.35355339 0.        ] 
```