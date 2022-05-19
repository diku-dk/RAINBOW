---
layout: page_documentation_single
title: "less"
filename: "vector3.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/vector3/less
---
## Purpose & Params
A less function for 3d vectors.Testing if a
    
    :param 1 name: Is a flat ```numpy``` array of size 3
    :param 2 name: Is a flat ```numpy``` array of size 3
    :return: boolean  

## Example
```python
    test_vec_1 = vec3.make(1,2,3)
    test_vec_2 = vec3.make(0,0,0)
    vec3.less(test_vec_2, test_vec_1)
```
output:
```bash
    true
```