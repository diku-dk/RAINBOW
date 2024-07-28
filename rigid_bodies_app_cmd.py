import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.geometry.surface_mesh as MESH
import rainbow.simulators.prox_rigid_bodies.api as API


def setup_scene(engine):
    V, T = MESH.create_box(10.0, 1.0, 10.0)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, 'ground', mesh)

    V, T = MESH.create_box(2.0, 2.0, 2.0)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, 'box', mesh)

    V, T = MESH.create_sphere(1.0, 16, 16)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, 'sphere', mesh)

    API.create_rigid_body(engine, 'A')
    API.connect_shape(engine, 'A', 'ground')
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
    API.set_position(engine, 'B', V3.make(-2.0, 2.0, 0.0), use_model_frame=True)
    API.set_velocity(engine, 'B', V3.make(0.0, 0.0, 0.0))
    API.set_spin(engine, 'B', V3.make(0.0, 0.0, 0.0))

    API.create_rigid_body(engine, 'C')
    API.connect_shape(engine, 'C', 'box')
    API.set_mass_properties(engine, 'C', 1.0)
    API.set_orientation(engine, 'C', Q.identity(), use_model_frame=True)
    API.set_position(engine, 'C', V3.make(2.0, 2.0, 0.0), use_model_frame=True)
    API.set_velocity(engine, 'C', V3.make(0.0, 0.0, 0.0))
    API.set_spin(engine, 'C', V3.make(0.0, 0.0, 0.0))

    API.create_gravity_force(engine,"earth-like", 10.0, V3.j())
    API.create_damping_force(engine, "windy-like", 0.001, 0.001)

    API.connect_force(engine, "A", "earth-like")
    API.connect_force(engine, "B", "earth-like")
    API.connect_force(engine, "C", "earth-like")

    API.connect_force(engine, "A", "windy-like")
    API.connect_force(engine, "B", "windy-like")
    API.connect_force(engine, "C", "windy-like")


def simulate(engine):
    dt = engine.params.time_step
    T = 0.5  # total time
    fps = 1.0 / dt
    steps = int(np.round(T * fps))
    for i in range(steps):
        API.simulate(engine, dt, profiling_on=True)
        print("simulated ", i, " steps, with ", len(engine.contact_points), " contact points")


def plot(engine):
    profile_data = API.get_log(engine)

    plt.figure(figsize=(12, 8))  # Increase figure size for better visibility
    ax = plt.subplot(111)

    ax.set_title('Convergence Rate of Engine Optimization', fontsize=16)  # Improved title
    ax.set_xlabel('Iterations', fontsize=14)  # More descriptive x-axis label
    ax.set_ylabel('Merit (Log Scale)', fontsize=14)  # More descriptive y-axis label

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)  # Improved grid
    ax.tick_params(axis='both', which='major', labelsize=12)  # Larger tick labels

    # Use a colormap for the lines
    colors = cm.viridis(np.linspace(0, 1, len(profile_data)))

    for i, data in enumerate(profile_data):
        if 'residuals' in data.keys():
            residuals = data['residuals']
            ax.semilogy(residuals,
                        color=colors[i],
                        linewidth=2
                        )
    plt.show()


def main():
    engine = API.create_engine()

    setup_scene(engine)

    simulate(engine)

    plot(engine)


if __name__ == '__main__':
    main()
