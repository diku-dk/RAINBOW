import numpy as np
import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.geometry.surface_mesh as MESH
import rainbow.simulators.prox_rigid_bodies.api as API
import matplotlib.pyplot as plt


def setup_scene(engine):
    V, T = MESH.create_box(10.0, 1.0, 10.0)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, 'box', mesh)

    V, T = MESH.create_sphere(1.0, 16, 16)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, 'sphere', mesh)

    API.create_rigid_body(engine, 'A')
    API.connect_shape(engine, 'A', 'box')
    API.set_mass_properties(engine, 'A', 1.0)
    API.set_orientation(engine, 'A', Q.identity(), use_model_frame=True)
    API.set_position(engine, 'A', V3.make(0.0, 0.0, 0.0), use_model_frame=True)
    API.set_velocity(engine, 'A', V3.make(0.0, 0.0, 0.0))
    API.set_spin(engine, 'A', V3.make(0.0, 0.0, 0.0))
    API.set_body_type(engine, 'A', 'fixed')

    API.create_rigid_body(engine, 'B')
    API.connect_shape(engine, 'B', 'sphere')
    API.set_mass_properties(engine, 'B', 1.0)
    API.set_orientation(engine, 'B', Q.identity(), use_model_frame=True)
    API.set_position(engine, 'B', V3.make(0.0, 2.0, 0.0), use_model_frame=True)
    API.set_velocity(engine, 'B', V3.make(0.0, 0.0, 0.0))
    API.set_spin(engine, 'B', V3.make(0.0, 0.0, 0.0))

    API.create_gravity_force(engine,"earthish", 10.0, V3.j() )
    API.create_damping_force(engine, "windish", 0.001, 0.001)

    API.connect_force(engine, "A", "earthish")
    API.connect_force(engine, "B", "earthish")
    API.connect_force(engine, "A", "windish")
    API.connect_force(engine, "B", "windish")


def simulate(engine):
    dt = engine.params.time_step
    T = 0.5  # total time
    fps = 1.0 / dt
    steps = int(np.round(T * fps))
    for i in range(steps):
        API.simulate(engine, dt, debug_on=True)
        print("simulated ", i, " steps, with ", len(engine.contact_points), " contact points")


def plot(engine):
    stats = API.get_log(engine)
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_title('Converegence rates')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Merit')
    plt.grid(True)
    for i in range(len(stats)):
        data = stats[i]
        if 'residuals' in data.keys():
            residuals = data['residuals']
            reject = data['reject']
            ax.plot(residuals[np.where(reject == False)])
    plt.show()


def main():
    engine = API.create_engine()

    setup_scene(engine)

    simulate(engine)

    plot(engine)


if __name__ == '__main__':
    main()
