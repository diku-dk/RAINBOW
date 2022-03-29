---
layout: page_documentation_single
title: "concat"
filename: "coordsys.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/coordsys/concat
---
## Purpose & Params
Concat two coordinate transformations. 

## Example
```python
    x,y,z          = 1,0,0
    radians_1      = np.pi 
    radians_2      = np.pi / 2
    axis_1         = np.array([0,1,0])
    axis_2         = np.array([0,0,1])
    r              = vec3.make(x,y,z)
    q_1            = quat.Ru(radians_1, axis_1)
    q_2            = quat.Ru(radians_2, axis_2)
    coordsys_1     = cs.make(r,q_1)
    coordsys_2     = cs.make(r,q_2)
    identity_basis = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ])
    concat_transform = cs.concat(coordsys_1, coordsys_2)
    print(f"Concat transformation r: {concat_transform.r}")
    print(f"q: {concat_transform.q}")
```

```bash
Concat transformation r: [ 0.0   0.0 -1e-16]
                      q: [ 4e-17 0.7    0.7 4e-17]
```