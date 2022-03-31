---
layout: page_documentation_single
title: "make_euler_xyz_matrix"
filename: "euler.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/euler/make_euler_xyz_matrix
math: http://latex.codecogs.com/svg.latex? 
---
## Find the angle from a euler rotation matrix
This function assumes that the order of the rotation is R(x)R(y)R(z). 
The implementation is inspired by section 2.6 in [D. Everly](https://www.geometrictools.com/Documentation/EulerAngles.pdf).


## Example 
First import the module
``` 
    import isl.math.euler as eu
```
Then
```
    radians_x = -1.2 
    radians_y = np.pi / 2 
    radians_z = 1.2 
    matrix         = np.dot(mat3.Rx(radians_z), np.dot(mat3.Ry(radians_y), mat3.Rz(radians_x)))
    Euler          = eu.make_euler_xyz_from_matrix(matrix)
    Euler.alpha, Euler.beta, Euler.gamma
```

Output
```
array([-0.65208111, -0.62503103,  0.42909955])
```