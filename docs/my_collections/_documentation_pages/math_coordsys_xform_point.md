---
layout: page_documentation_single
title: "xfrom_point"
filename: "coordsys.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/coordsys/xform_point
---
## Purpose & Params
The purpose of xfrom_point transforms a point to another basis. The first argument is the basis and the second point is the point.        

## Example
```python
    x,y,z              = 0,-2,0
    radians            = np.pi / 2         # rotate  90 degrees
    axis               = np.array([0,0,1])
    r                  = vec3.make(x,y,z)
    q                  = quat.Ru(radians, axis)
    coordsys           = cs.make(r,q)
    point              = np.array([0,2,0])
    transformed_point  = cs.xform_point(coordsys, point)
    print(f'Initial Point: {point}')
    print(f'Transformed Point: {transformed_point}')
```

```bash
    Initial Point: [0 2 0]
    Transformed Point: [-2. -2.  0.]
```