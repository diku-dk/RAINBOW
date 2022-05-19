---
layout: page_documentation_single
title: "polar_decomposition"
filename: "matrix3.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/matrix3/polar_decomposition
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
Polar Decomposition of matrix A (as described by Etzmuss et. al in ``A Fast Finite Solution for Cloth Modelling'')
### Properties 
![]({{page.math}} R \cdot R^T = I)

![]({{page.math}} det(R)      = 1)

## Example
```python
    M = np.random.rand(3,3)
    R, S = mat3.polar_decomposition(M)

    print("------ M -----")
    print(M)
    print("----- S^T -----")
    print(np.transpose(S))
    print(np.dot(R, np.transpose(R)))
    print(np.dot(R,S))
```
output:
```bash
   ------ M ------
   [[0.67751452 0.29537368 0.81107846]
    [0.7250961  0.16420613 0.87102864]
    [0.71635474 0.0468772  0.53182239]]
   ----- S^T -----
   [[0.92180171 0.14036562 0.79280104]
    [0.14036562 0.18668312 0.24870392]
    [0.79280104 0.24870392 1.00448349]]
   --- R * R^T ---
   [[ 1.00000000e+00  5.27355937e-15 -7.71605002e-15]
    [ 5.27355937e-15  1.00000000e+00 -5.82867088e-16]
    [-7.71605002e-15 -5.82867088e-16  1.00000000e+00]]
   ---- R * S ----
   [[0.67751452 0.29537368 0.81107846]
    [0.7250961  0.16420613 0.87102864]
    [0.71635474 0.0468772  0.53182239]]
```