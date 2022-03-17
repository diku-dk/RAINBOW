---
layout: page_documentation_single
title: "polynomial"
filename: "polynomial.py"
folder_dir: "isl/math"
author: Kenny Erleben
maintainer: Kenny Erleben
permalink: my_collections/documentation_pages/math/polynomial
math: http://latex.codecogs.com/svg.latex?
---
## Purpose
This module provides support for multivariate polynomials. Focus is on one-, two- and three-dimensional polynomials. Particular how to find an interpolating multivariate polynomial, evaluate its value at given points and finding the derivative of the polynomial. Below is an example usage of the module.

## Example
```python
import numpy as np
import isl.math.polynomial as POLY


x = np.rand.random(10,2)*10
v = np.rand.random(10,3)
order = 2
P0, P1, P2 = POLY.interpolate(order, x, v)

print(P1(x[10]))

dPdx, dPdy = POLY.derivative(P1)

print(dPdx)

```
output:
```bash

1 + x + y
```