---
layout: page_documentation_single
title: "Exponential Map Guide"
filename: "expmap.py"
folder_dir: "isl/math/"
author: Kenny Erleben
maintainer: DIKU
permalink: my_collections/documentation_pages/math/expmap/exp_guide
math: http://latex.codecogs.com/svg.latex? 
---
## The Exponential Map
The purpose of the exomap.py module is to make exponential mapping possible. 
We used the code examples and theory from the paper of [F. Sebastian Grossia](https://www.cs.cmu.edu/~spiff/moedit99/expmap.pdf) to implement this module.
    
### Definition
Let ![]({{page.math}} \Theta = |v|), where Theta is in radians and ![]({{page.math}} \hat{\Theta} = v/|v|) then

![]({{page.math}} q = e^{v} = [cos(\frac{1}{2}\Theta ), \frac{sin(\frac{1}{2} \Theta )}{\Theta} v]^T)

This module helps parameterize rotations matrices in group [SO(3)](https://en.wikipedia.org/wiki/3D_rotation_group). Rotations with these parameters avoid 
singularities by converting every rotation to be in the range of 0 and ![]({{page.math}} \pi).

## Examples
First load the module
```python
    import isl.math.expmap
```
### Example 1
This example shows how to initialize a rotation matrix from v. 
```python
        # Rotates with pi/2
        v_rot_const = (np.pi/2) * (1/(np.sqrt(3)))
        v           = vec3.make(v_rot_const,v_rot_const,v_rot_const)
        exp_map     = exp(v)
        to_matrix(exp_map)
```
Output
```
array([[ 0.84066355, -0.21922386,  0.49520268],
       [ 0.49520268,  0.6813271 , -0.5390433 ],
       [-0.21922386,  0.69837975,  0.6813271 ]])
```
### Example 2
Finding the inverse of ***exp_map***
```python
    log(exp_map), v
```
Output
```
(array([0.90689968, 0.90689968, 0.90689968]),
 array([0.90689968, 0.90689968, 0.90689968]))
```

### Example 3
If you want to rotate with more than pi you need to reparmeterize.
```python
    # Rotate with 1.5 * pi
    v_rot_const = (1.5*np.pi) * (1/(np.sqrt(3)))
    v           = vec3.make(v_rot_const,v_rot_const,v_rot_const)
    v           = reparameterization(v)
    exp_map     = exp(v)
    print(f' Theta_hat: {2*np.pi -  vec3.norm(v)}, Theta: {1.5*np.pi}')
```
Output
```
Theta_hat: 4.7123889803846915, Theta: 4.71238898038469
```

>### Example 4
Find the derivative of v given an angle omega (See paper section 4.1).
```python
    rot_const_omega     =  np.pi + np.pi / 4
    rot_const_v         =  np.pi + np.pi / 2 
    quad_const          = 1/np.sqrt(3)
    rot_1               = rot_const_omega * quad_const
    rot_2               = rot_const_v * quad_const
    omega     = vec3.make(rot_1,rot_1,rot_1)
    v         = vec3.make(rot_2,rot_2,rot_2)
    dvdt(omega, v)
```
Output
```
array([2.26724921, 2.26724921, 2.26724921])
```

### Example 5
Finding the derivative of a rotation matrix with respect to a axis ***i*** (See section 3.1 in the paper).
Let i = 0 meaning the first axis:
```python
    const1     = np.pi +  np.pi / 4
    quad_const = 1/np.sqrt(3)
    rot        = const1 * quad_const
    v          = vec3.make(rot,rot,rot)
    v          = reparameterization(v)
    dRdv_i(v, 0)
```
Output
```
    array([[-0.28556917, -0.35808005, -0.3111484 ],
           [-1.02955486, -0.64055624, -0.23988447],
           [-0.23988447, -0.3111484 , -0.64055624]])
```
