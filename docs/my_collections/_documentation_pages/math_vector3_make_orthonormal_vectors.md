---
layout: page_documentation_single
title: "make_orthonomal_vectors"
filename: "vector3.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/vector3/make_orthonomal_vectors
---
## Purpose & Params
The purpose of this function is the create the tanget and binormal of the
unit vector n
[0.,0.,0.]
    :return: 
        tangent  vector, numpy float64 array
        binomal  vector, numpy float64 array    
        n        vector, numpy float64 array

## Example
```python
    norm_vector = vec3.unit(vector)
    tangent, binormal, n = vec3.make_orthonormal_vectors(norm_vector)
    expected = n
    actual   = np.cross(tangent, binormal)

    print(actual == expected)
```
output:
```bash
    true
```