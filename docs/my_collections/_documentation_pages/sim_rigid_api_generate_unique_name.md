---
layout: page_documentation_single
title: "generate_unique_name"
filename: "api.py"
folder_dir: "isl/simulators/prox_rigid_bodies/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/simulations/prox_rigid_bodies/api/generate_unique_name
---
## Purpose & Params
This function helps to generate unique names, such that one can always locate objects based on name only.

    :param name:   The original name wanted.
    :return:       pre and post appended name string that makes name unique.


## Example
```python
import isl.simulators.prox_rigid_bodies.api as api

test_name   = "Name"
unique_name_1 = api.generate_unique_name(test_name)
unique_name_2 = api.generate_unique_name(test_name)

unique_name_1 == unique_name_2
```
output:
```bash
false
```