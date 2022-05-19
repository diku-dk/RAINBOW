---
layout: page_documentation_single
title: "prod"
filename: "quaternion.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/quaternion/prod
math: http://latex.codecogs.com/svg.latex? 
---
## Purpose & Params
The purpose of this function is to make a Quaternion product

```
    Two queaterions q = [s,v] and q' = [s',v'] are multiplied with each other as follows
```
![]({{page.math}}qq' - v \cdot v', sv' + s'v + v \times v)


```
    :param: make 
    :return: [] numpy.float64, length 4  
```

## Example
```python
    quaternion   = quat.make(1,2,3,4)
    print(test_vec_1)
```
output:
```bash
   [1,2,3,4] 
```