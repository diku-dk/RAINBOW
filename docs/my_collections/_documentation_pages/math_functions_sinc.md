---
layout: page_documentation_single
title: "sinc"
filename: "functions.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/functions/sinc
math: http://latex.codecogs.com/svg.latex? 
---
## Compute Sinc function
A robust implementation of the sinc functions. 
The implementation of this method was greatly inspired by the
one in Open Dynamics Engine v. 0.039

## Future implementation
Support for array input will be implemented.

## Example 
First import the module
``` 
    import isl.math.functions as func
```
Then
```
    var = 2
    func.sinc(var)
```
Output
```
    0.45464871341284085 
```