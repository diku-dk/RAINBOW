---
layout: default
title: "Examples"
permalink: /examples/
---
## Imports
First import the modules
{% include codeHeader.html %}
```python
import numpy as np
import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.simulators.prox_rigid_bodies.api as API
import rainbow.simulators.prox_rigid_bodies.procedural as PROC
import rainbow.util.viewer as VIEWER
import time
from rainbow.geometry.surface_mesh import create_sphere, create_box
```



## Examples 1 - Get started
Let's make a box rotate 360 degrees by setting its angular velocity to 2 pi and run the simulation for a second.  

Start by creating an environment. In this case, we create a box with dimensions 5., 5. and 10. 
{% include codeHeader.html %}
```python
engine = API.Engine()

API.create_rigid_body(engine,'box_body')

V, T   = create_box(5.,5., 10.) 

mesh = API.create_mesh(V, T)
API.create_shape(engine, 'box_shape', mesh )
API.connect_shape(engine, 'box_body', 'box_shape')
```

Let's set the mass and the angular velocity. 

{% include codeHeader.html %}
```python
API.set_mass_properties(engine,'box_body', 1)
API.set_spin(engine, 'box_body', V3.make(2.*np.pi, 0., 0.))
```

Next, to view your setup, use the viewer class.
{% include codeHeader.html %}
```python
viewer = VIEWER.Viewer()
for body in engine.bodies.values():
    opacity = 0.5
    color = V3.make(1.0,0.1,0.1)
    viewer.create_mesh(body.name, body.shape.mesh.V, body.shape.mesh.T, color, opacity)
    viewer.place_mesh(body.name, body.r, body.q)
    
viewer.show()
```
Finally, simulate over time by creating a simulation loop.
{% include codeHeader.html %}
```python
def simulation(viewer, engine, monitor=True) -> None:
    dt = engine.params.time_step
    T  = engine.params.total_time 
    fps = 1.0/dt
    steps = int(np.round(T*fps))
    for i in range(steps):
        for body in engine.bodies.values():
            viewer.place_mesh(body.name, body.r, body.q)
        API.simulate(engine, dt, monitor)

engine.params.total_time = 1 #sec       
simulation(viewer, engine, True)
```

# Example 2 - Interactions
Let's create a simulation with a simple interaction. Thus we initialize two spheres. Next, we force the spheres to collide by setting their initial velocity opposite each other.  

First, create an environment as before. 

{% include codeHeader.html %}
```python
engine = API.Engine()
API.create_rigid_body(engine,'sphere_body_1')
API.create_rigid_body(engine,'sphere_body_2')

V_1, T_1 = create_sphere(5.0,16,16)
V_2, T_2 = create_sphere(5.0,16,16)

mesh_1 = API.create_mesh(V_1, T_1)
mesh_2 = API.create_mesh(V_2, T_2)

API.create_shape(engine, 'sphere_shape_1', mesh_1)
API.create_shape(engine, 'sphere_shape_2', mesh_2)

API.connect_shape(engine, 'sphere_body_1', 'sphere_shape_1')
API.connect_shape(engine, 'sphere_body_2', 'sphere_shape_2')

API.set_mass_properties(engine,'sphere_body_1', 1)
API.set_mass_properties(engine,'sphere_body_2', 1)

API.set_position(engine, 'sphere_body_1', V3.make( 55.0, 0.0,0.0), use_model_frame=True)
API.set_position(engine, 'sphere_body_2', V3.make(-55.0, 0.0,0.0), use_model_frame=True)

API.set_velocity(engine, 'sphere_body_1', V3.make( -100., 0., 0.))
API.set_velocity(engine, 'sphere_body_2', V3.make(  100., 0., 0.))
```

The materials of the spheres have an elasticity factor of 0.5. (An object has "default" as material as default.)

{% include codeHeader.html %}
```python
test_material = "material"
API.create_surfaces_interaction(engine, test_material, test_material, 0.5, V3.make(np.inf, np.inf, np.inf))
API.set_body_material(engine, 'sphere_body_1', test_material)
API.set_body_material(engine, 'sphere_body_2', test_material)
engine.params.use_bounce = True
```

Simulate as before.
{% include codeHeader.html %}
```python
def simulation(viewer, engine, monitor=True) -> None:
    dt = engine.params.time_step
    T  = engine.params.total_time 
    fps = 1.0/dt
    steps = int(np.round(T*fps))
    for i in range(0, steps+1):
        for body in engine.bodies.values():
            viewer.place_mesh(body.name, body.r, body.q)
        API.simulate(engine, dt, monitor)

engine.params.total_time = 1 #sec 
engine.params.time_step  = 0.01
simulation(viewer, engine, True)
```

# Example 3 - Use the force
This tutorial shows how to add forces such as gravitational force. 

First, initialize the environment. 
{% include codeHeader.html %}
```python
engine = API.Engine()
PROC.create_ground(engine, V3.zero(), Q.Rx(0.3*np.pi), density=1000.0, material_name='default');

API.create_rigid_body(engine,'sphere_body')

V_1, T_1 = create_sphere(5.0,16,16)

mesh_1 = API.create_mesh(V_1, T_1)

API.create_shape(engine, 'sphere_shape', mesh_1)

API.connect_shape(engine, 'sphere_body', 'sphere_shape')

API.set_mass_properties(engine,'sphere_body', 1/504) # Creates an mass of 1

v0  = V3.make(10.0, 10.,0 )
x0  = V3.make( -40.0, 30.0,0.0)
direction_vector = V3.make(0.,1.,0.)
gravity_force     = 0.1

API.set_position(engine, 'sphere_body', x0, use_model_frame=True)
API.set_velocity(engine, 'sphere_body', v0)

API.create_gravity_force(engine, 'gravity_f', gravity_force, direction_vector)
API.connect_force(engine, 'sphere_body', 'gravity_f')
```

Simulate,
{% include codeHeader.html %}
```python
def simulation(viewer, engine, monitor=True) -> None:
    dt = engine.params.time_step
    T  = engine.params.total_time 
    fps = 1.0/dt
    steps = int(np.round(T*fps))
    for i in range(steps):
        for body in engine.bodies.values():
            viewer.place_mesh(body.name, body.r, body.q)
        API.simulate(engine, dt, monitor)

engine.params.total_time = 100 #sec
engine.params.time_step  = 0.1
simulation(viewer, engine, True)
```

# Example 4 - Friction
Let's make a sliding box to illustrate how friction. 
The environment,
{% include codeHeader.html %}
```python
engine = API.Engine()

mu_x     = -0.1*np.pi
mu_y     = 0.0
mu_z     = 0.0

v_mu    = V3.make(mu_x, mu_y, mu_z)
q_mu   = Q.Rx(mu_x)
size = 6.0 

test_material = "material"
API.create_surfaces_interaction(engine, test_material, test_material, 1, np.arctan(V3.make((0.09*np.pi), np.inf, np.inf)))

PROC.create_ground(engine, V3.zero(), q_mu, density=1000.0, material_name=test_material)
API.create_rigid_body(engine,'sphere_body')

V_1, T_1 = create_box(size, size, size)

mesh_1 = API.create_mesh(V_1, T_1)

API.create_shape(engine, 'sphere_shape', mesh_1)

API.connect_shape(engine, 'sphere_body', 'sphere_shape')

rx  = Q.Rx(mu_x)
pos = Q.rotate(rx, V3.make(0.0, size / 2,0.0))

direction_vector = V3.make(0.,1.,0.)
gravity_force     = 0.1

API.set_position(engine, 'sphere_body', pos, use_model_frame=True)
API.set_orientation(engine, 'sphere_body', rx, use_model_frame=True)
API.set_mass_properties(engine,'sphere_body', 1)

API.create_gravity_force(engine, 'gravity_f', gravity_force, direction_vector)
API.connect_force(engine, 'sphere_body', 'gravity_f')


API.set_body_material(engine, 'sphere_body', test_material)
engine.params.use_bounce = True
```
Simulate,
{% include codeHeader.html %}
```python
def simulation(viewer, engine, monitor=True) -> None:
    dt = engine.params.time_step
    T  = engine.params.total_time 
    fps = 1.0/dt
    steps = int(np.round(T*fps))
    for i in range(steps):
        for body in engine.bodies.values():
            viewer.place_mesh(body.name, body.r, body.q)
        API.simulate(engine, dt, monitor)
engine.params.total_time = 100   
engine.params.time_step  = 0.1
simulation(viewer, engine, True)
```
### Analyse the friction simulation

Use the get_log function:
{% include codeHeader.html %}
```python
stats = API.get_log(engine)
```
Plot convergence.
{% include codeHeader.html %}
```python
import matplotlib.pyplot as plt
colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
          '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
          '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
          '#000075', '#808080', '#ffffff', '#000000']
fig = plt.figure()
ax = plt.subplot(111)
ax.set_title('Convergence rates')
ax.set_xlabel('Iterations')
ax.set_ylabel('Merit')
plt.grid(True)
for i in range(len(stats)):
    data = stats[i]
    if 'residuals' in data.keys():
        residuals = data['residuals']
        reject = data['reject']
        ax.plot( residuals[np.where(reject==False)])
plt.show()
```
Profile the simulation.
{% include codeHeader.html %}
```python
time_update_bvh = [ stats[i]['update_bvh'] for i in range(len(stats)) ]
time_narrow_phase = [ stats[i]['narrow_phase'] for i in range(len(stats)) ]
time_contact_determination = [ stats[i]['contact_determination'] for i in range(len(stats)) ]
time_contact_point_reduction = [ stats[i]['contact_point_reduction'] for i in range(len(stats)) ]
time_collision_detection = [ stats[i]['collision_detection_time'] for i in range(len(stats)) ]

time_stepper = [ stats[i]['stepper_time'] for i in range(len(stats)) ]


fig = plt.figure()
ax = plt.subplot(111)
ax.set_title('Profiling Timings')
ax.set_xlabel('Step')
ax.set_ylabel('Time [s]')
plt.grid(True)
ax.plot(time_update_bvh, label='Update bvh', color=colors[6])
ax.plot(time_narrow_phase, label='Narrow phase', color=colors[7])
ax.plot(time_contact_determination, label='Contact determination', color=colors[8])
ax.plot(time_contact_point_reduction, label='Contact reduction', color=colors[9])
ax.plot(time_collision_detection, label='Collision Detection', color=colors[10])
ax.plot(time_stepper, label='Stepper', color=colors[11])
ax.legend()
plt.show()

number_of_overlaps = [ stats[i]['number_of_overlaps'] for i in range(1, len(stats)) ]
step_sizes = [ stats[i]['dt'] for i in range(1, len(stats)) ]
number_of_contact_points = [ stats[i]['contact_points'] for i in range(1, len(stats)) ]
penetrations = [ stats[i]['max_penetration'] for i in range(1, len(stats)) ]

fig = plt.figure()
ax = plt.subplot(111)
ax.set_title('Profiling data')
ax.set_xlabel('Step')
ax.set_ylabel('Value')
plt.grid(True)
ax.plot(number_of_overlaps, label='Overlaps', color=colors[0])
ax.plot(step_sizes, label='Stepsize', color=colors[1])
ax.plot(number_of_contact_points, label='Contacts', color=colors[2])
ax.plot(penetrations, label='Penetrations', color=colors[6])
ax.legend()
plt.show()
```
Show the potential energy and the kinetic energy.
{% include codeHeader.html %}
```python
kinetic_energy = [ stats[i]['kinetic_energy'] for i in range(len(stats)) ]
potential_energy = [ stats[i]['potential_energy'] for i in range(len(stats)) ]

fig = plt.figure()
ax = plt.subplot(111)
ax.set_title('Energy Plots')
ax.set_xlabel('Step')
ax.set_ylabel('Value')
plt.grid(True)
ax.plot(kinetic_energy, label='Kinetic Energy', color=colors[4])
ax.plot(potential_energy, label='Potential Energy', color=colors[5])
ax.legend()
plt.show()
```
# Example 5 - Advanced Shapes
Let's make a sliding box to illustrate how friction. In the last example, we create a more advanced shape using the procedural module. 
{% include codeHeader.html %}
```python
engine = API.Engine()
PROC.create_ground(engine, V3.zero(), Q.identity(), density=1000.0, material_name='default');
PROC.create_dome(engine,
                 r = V3.zero(),
                 q = Q.identity(),
                 outer_radius = 5.0,
                 inner_radius = 4.0,
                 layers = 4,
                 segments = 11,
                 density = 1.0,
                 material_name = 'default'
                )
```
Simulate
{% include codeHeader.html %}
```python
def simulation(viewer, engine, monitor=True) -> None:
    dt = engine.params.time_step
    T  = engine.params.total_time 
    fps = 1.0/dt
    steps = int(np.round(T*fps))
    for i in range(steps):
        for body in engine.bodies.values():
            viewer.place_mesh(body.name, body.r, body.q)
        API.simulate(engine, dt, monitor)
engine.params.total_time = 100   
engine.params.time_step  = 0.1
simulation(viewer, engine, True)

```