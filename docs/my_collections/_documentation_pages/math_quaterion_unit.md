---
layout: page_documentation_single
title: "unit"
filename: "quaternion.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/quaternion/unit
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
The norm (or length) of a quaternion is defined by

![]({{page.math}} \|q\| = \sqrt{q \cdot q})


A quaternion, with norm equal to 1, is called a unit quaterion. If we have the unit quaternion,

![]({{page.math}} \|n\| = \[cos(\theta), n \cdot sin(\theta)\])

the it represents a rotation of 2 theta radians around the normalized axis n.


## Example
```python
    quaternion      = quat.make(1,2,3,4)
    quaternion_unit = quat.unit(quaternion)
    print(quaternion_unit)
```
output:
```bash
   array([0.18257419, 0.36514837, 0.54772256, 0.73029674]) 
```