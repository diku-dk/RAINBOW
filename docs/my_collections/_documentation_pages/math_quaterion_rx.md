---
layout: page_documentation_single
title: "Rx"
filename: "quaternion.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/quaternion/Rx
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
Given a radian, the function returns a quaternion rotation on the x-axis
given the orientation. Get more information about rotation on the quaternion rotation documentation webpage.

## Example
```python
    radian = (0.5 * np.pi) / 2
    print(quat.Rx(radian))
```
output:
```bash
   [0.92387953 0.38268343 0.         0.        ]
```