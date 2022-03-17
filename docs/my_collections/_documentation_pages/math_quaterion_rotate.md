---
layout: page_documentation_single
title: "rotate"
filename: "quaternion.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/quaternion/rotate
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
Given ![]({{page.math}} p = \[0,r\]) and a unit quaternion 
![]({{page.math}} q = \[cos(\theta),n \cdot sin(\theta)\]), then the rotation of p is given by

![]({{page.math}} p' = qpq*)

the it represents a rotation of ![]({{page.math}} 2 \theta) radians around the axis n

## Example
```python
    r      = np.array([1,0,0])
    n      = np.array([0,1,0])
    radian = (0.5 * np.pi) / 2
    q      = np.array([np.cos(radian),n[0]*np.sin(radian),n[1]*np.sin(radian),n[2]*np.sin(radian)], dtype=np.float64)
    print(quat.rotate(q, r))
```
output:
```bash
   array([0.0,0.0,-1.0])
```