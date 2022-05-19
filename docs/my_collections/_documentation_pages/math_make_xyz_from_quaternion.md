---
layout: page_documentation_single
title: "make_euler_xyz_quaternion"
filename: "euler.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/euler/make_xyz_from_quaternion
math: http://latex.codecogs.com/svg.latex? 
---
## Find the angle from a euler 
This function assumes that the order of the rotation is R(x)R(y)R(z).

### Definition
Given a unit quaternion ![]({{page.math}} q = [s, (x,y,z)]), the corresponding rotation matrix is

![]({{page.math}}M= \begin{pmatrix}
                    1-2(y^2+z^2) & 2xy-2sz      & 2sy+2xz     \\
                    2xy+sy       & 1-2(x^2+z^2) & -2sx + syz  \\
                    2sy+2xz      & 2sx+2yz      & 1-2(x^2+y^2)
                  \end{pmatrix} )

More information can be found in page 602 in physics-based animation by Erleben, 
Sporring, Heniksen and Dohlmann. 
The angles can be found from the above rotation matrix
## Example
```
    radians_x =  0.4 
    radians_y =  1.1
    radians_z = -1.3
    matrix         = np.dot(mat3.Rz(radians_z), np.dot(mat3.Ry(radians_y), mat3.Rx(radians_x)))
    q              = quat.from_matrix(matrix)
    Euler          = eu.make_euler_xyz_from_quaternion(q)
    Euler.alpha, Euler.beta, Euler.gamma
```
Output
```
    (0.4000000000000001, 1.1, -1.3)
```
