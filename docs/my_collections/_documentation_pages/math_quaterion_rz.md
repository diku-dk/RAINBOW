---
layout: page_documentation_single
title: "Rz"
filename: "quaternion.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/quaternion/Rz
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
Given a radian, the function returns a quaternion rotation on the z-axis
given the orientation. Get more information about rotation on the quaternion rotation documentation webpage.

## Example
```python
    radian = (0.5 * np.pi) / 2
    print(quat.Rz(radian))
```
output:
```bash
   [0.92387953 0.         0.         0.38268343]
```