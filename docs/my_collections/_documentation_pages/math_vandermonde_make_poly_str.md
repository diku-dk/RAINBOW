---
layout: page_documentation_single
title: "make_poly_str"
filename: "vandermonde.py"
folder_dir: "isl/math"
author: Kenny Erleben
maintainer: Kenny Erleben
permalink: my_collections/documentation_pages/math/vandermonde/make_poly_str
math: http://latex.codecogs.com/svg.latex?
---
## Purpose & Params
This function creates a polynomial string containing all monomials upto the specified order.

```
:param: degree The desired degree of the polynomial being generated. 
:return: str   Latex string representation of the polynomial.  
```

## Example
```python
import isl.math.vandermonde as V

poly = V.make_poly_str(1)
print(poly)
```
output:
```bash
1 + x + y
```