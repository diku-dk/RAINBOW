---
layout: page_documentation_single
title: "from_string"
filename: "quaternion.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/quaternion/from_string
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
The purpose of this module is to convert string to operations. The convertible operations is: 

- "identity"        : Converts to the identity 
- "rx: < degree >"  : Converts to a quatertion rotating < degree > degrees    
                      around the x axis.
- "ry: < degree >"  : Converts to a quatoration rotating < degree > degrees    
                      around the y axis.
- "rz: < degree >"  : Converts to a quatertion rotating < degree > degrees    
                      around the z axis.
- "ru: < degree >: < axis >" : Converts to a quatertion< degree > degrees    
                      around the < axis > axis. 
- "[x1,x2,...xN]"   : Converts to an array of size N.    


## Example
```python
    import numpy as np
    degree  = "90"
    axis    = "[1,0,0]"
    quaternion = from_string(f"ru: {degree}:{axis}")
    print(quaternion)
```
output:
```bash
   array([0.70710678, 0.70710678, 0.        , 0.        ])
```