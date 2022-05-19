---
layout: page_documentation_single
title: "from_string"
filename: "vector3.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/vector3/from_string
---
## Purpose & Params
    Converts a string ot a numpy array

    :param: string 
    :return: numpy array of numpy type float64  

## Example
```python
    test_vec_1 = vec3.from_string("[1,2,3]")
    print(test_vec_1)
```
output:
```bash
    [1.,2.,3.]
```