---
layout: page_documentation_single
title: "from_vector3"
filename: "quaternion.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/quaternion/from_vector3
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
The purpose of this module is to convert a vector of length 3 to a quaternion. 


## Example
```python
    import numpy as np
    quat = quat.from_vector3(np.array([1,2,3]))
    print(quat)
```
output:
```bash
   array([0.0, 1.0, 2.0, 3.0])
```