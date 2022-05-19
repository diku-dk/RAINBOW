---
layout: page_documentation_single
title: "polar_decomposition_array"
filename: "matrix3.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/matrix3/polar_decomposition_array
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
Perform polar decomposition on multiple matrices. See more information about polar decomposition on the polar decomposition documentation page.

## Example
```python
    number_of_M = 2
    shape = (number_of_M,3,3)
    Ms = np.random.rand(shape[0], shape[1], shape[2])
    Rs, Ss = mat3.polar_decomposition_array(Ms)

    print("Ms")
    print(Ms)
    print("all r dot with transpose in Rs")
    print(np.array([np.dot(r, np.transpose(r)) for r in Rs]))
    print("all r dot s in Rs and Ss")
    print(np.array([np.dot(r,s) for r,s in zip(Rs,Ss)]))
```
output:
```bash
   Ms
[[[0.05823526 0.42753248 0.40590315]
  [0.39075253 0.48787175 0.16834629]
  [0.63097681 0.66230782 0.92624305]]

 [[0.28748076 0.26974706 0.16509425]
  [0.30101813 0.60598822 0.58438612]
  [0.41011178 0.59532267 0.7780367 ]]]
all r dot with transpose in Rs
[[[ 1.00000000e+00  3.35842465e-15  8.32667268e-16]
  [ 3.35842465e-15  1.00000000e+00 -8.32667268e-16]
  [ 8.32667268e-16 -8.32667268e-16  1.00000000e+00]]

 [[ 1.00000000e+00  1.70696790e-15 -2.33146835e-15]
  [ 1.70696790e-15  1.00000000e+00 -2.22044605e-15]
  [-2.33146835e-15 -2.22044605e-15  1.00000000e+00]]]
all r dot s in Rs and Ss
[[[0.05823526 0.42753248 0.40590315]
  [0.39075253 0.48787175 0.16834629]
  [0.63097681 0.66230782 0.92624305]]

 [[0.28748076 0.26974706 0.16509425]
  [0.30101813 0.60598822 0.58438612]
  [0.41011178 0.59532267 0.7780367 ]]]
```