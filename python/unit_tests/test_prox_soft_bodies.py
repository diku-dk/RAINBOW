import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.simulators.prox_soft_bodies.api as API
import rainbow.simulators.prox_soft_bodies.solver as SOLVER
import rainbow.simulators.prox_soft_bodies.mechanics as MECH
import rainbow.simulators.prox_soft_bodies.collision_detection as CD
import rainbow.simulators.prox_rigid_bodies.gauss_seidel as GS
import rainbow.geometry.volume_mesh as VM
import rainbow.math.vector3 as V3
import rainbow.math.matrix3 as M3
import numpy as np
import matplotlib.pyplot as pyplot

# 2022-04-10 Kenny TODO: Please re-implement into proper unit-test

class TestBlackBoxSimulator:

    def __init__(self):
        #self.test_contraction_setup()
        #self.test_gravity_setup()
        self.test_traction_setup()

    def test_contraction_setup(self):
        V, T = VM.create_beam(2, 2, 2, 1.0, 1.0, 1.0)
        engine = API.Engine()

        API.create_soft_body(engine, 'cube', V, T)
        API.set_type(engine, 'cube', 'Free')

        API.create_material(engine, 'steel')
        API.set_elasticity(engine, 'steel', 100.0, 0.45)
        API.set_mass_density(engine, 'steel', 1.0)
        API.set_constitutive_model(engine, 'steel', API.SVK)
        API.set_viscosity(engine, 'steel', 10.0)
        API.create_surfaces_interaction(engine,'steel','steel', 0.5)
        API.set_gravity(engine, 'cube', (0, 0, 0))  #No gravity forces

        API.set_material(engine, 'cube', 'steel')

        # Create an initial deformation, so beam will contract it self along x-axis.
        x = API.get_material_coordinates(engine, 'cube')
        x[:, 0] = 1.1*x[:, 0]
        API.set_spatial_coordinates(engine, 'cube', x)

        print('---- before stepper ---------------')
        for body in engine.bodies.values():
            print("velocities", body.u)
            print("spatial coords = ", body.x)
            print("elastic forces = ", body.Fe)

        for i in range(1000):
            SOLVER.stepper(0.01, engine, debug_on=True)

        print('---- after stepper --------------')
        for body in engine.bodies.values():
            print("velocities", body.u)                   # Should be close zero
            print("spatial coords = ", body.x)            # Should be close to material space
            print("elastic forces = ", body.Fe)           # Should be close zero

    def test_gravity_setup(self):
        V, T = VM.create_beam(2, 2, 2, 1.0, 1.0, 1.0)
        engine = API.Engine()

        API.create_soft_body(engine, 'cube', V, T)
        API.set_type(engine, 'cube', 'Free')

        API.create_material(engine, 'steel')
        API.set_elasticity(engine, 'steel', 100.0, 0.45)
        API.set_mass_density(engine, 'steel', 1.0)
        API.set_constitutive_model(engine, 'steel', API.SVK)
        API.create_surfaces_interaction(engine, 'steel', 'steel', 0.5)
        API.set_gravity(engine, 'cube', (0, 0, -10))
        API.set_viscosity(engine, 'steel', 0.0)
        API.set_material(engine, 'cube', 'steel')

        print('---- before stepper ---------------')
        for body in engine.bodies.values():
            print("velocities", body.u)
            print("spatial coords = ", body.x)

        for i in range(1000):
            SOLVER.stepper(0.001, engine, debug_on=True)

        print('---- after stepper --------------')
        for body in engine.bodies.values():
            print("velocities", body.u)                   # Should be close to 0,0,-10
            print("spatial coords = ", body.x)            # Should be close to displaced by (0,0,-5)
            print("elastic forces = ", body.Fe)           # Should be zero
            print("external forces = ", body.Fext)        # Should be non-zero

    def test_traction_setup(self):
        def phi(x):
            return x[0] + 0.49

        V, T = VM.create_beam(2, 2, 2, 1.0, 1.0, 1.0)
        engine = API.Engine()

        API.create_soft_body(engine, 'cube', V, T)
        API.set_type(engine, 'cube', 'Free')
        API.create_material(engine, 'steel')
        API.set_elasticity(engine, 'steel', 100.0, 0.45)
        API.set_mass_density(engine, 'steel', 1.0)
        API.set_constitutive_model(engine, 'steel', API.SVK)
        API.create_surfaces_interaction(engine, 'steel', 'steel', 0.5)
        API.set_gravity(engine, 'cube', (0, 0, 0))
        API.set_viscosity(engine, 'steel', 0.0)
        API.set_material(engine, 'cube', 'steel')

        load = np.array([1., 0., 0.])
        API.create_traction_conditions(engine, 'cube', phi, load)

        print('---- before stepper ---------------')
        for body in engine.bodies.values():
            print("velocities", body.u)
            print("spatial coords = ", body.x)

        for i in range(1000):
            SOLVER.stepper(0.001, engine, debug_on=True)

        print('---- after stepper --------------')
        for body in engine.bodies.values():
            print("velocities", body.u)                   # Should be non-zero
            print("spatial coords = ", body.x)            # Should be  displaced by (?,0,0)
            print("elastic forces = ", body.Ft)           # Should be non-zero
            print("elastic forces = ", body.Fe)           # Should be zero



class TestSoftBodySimulator:

    def __init__(self):
        #self.test_one()
        #self.test_two()
        #self.test_traction_force()
        #self.test_three()
        self.test_forces()
        #self.test_solver()

    def test_one(self):
        print('running test one')
        V, T = VM.create_beam(2, 2, 2, 2.0, 1.0, 1.0)

        engine = API.Engine()
        API.create_soft_body(engine, 'test', V, T)
        API.set_velocity(engine, 'test', (0, 0, 0))
        API.set_type(engine, 'test', 'FrEE')
        API.set_type(engine, 'test', 'fixED')

        API.create_material(engine, 'steel')
        API.set_elasticity(engine, 'steel', 1.0, 0.3)
        API.set_viscosity(engine, 'steel', 1.0)
        API.set_mass_density(engine, 'steel', 1.0)
        API.set_constitutive_model(engine, 'steel', API.SVK)
        API.set_gravity(engine, 'test', (0, 0, -1))
        API.set_material(engine, 'test', 'steel')

        # API.create_surfaces_interaction(engine, 'steel', 'steel', 0.5)
        API.create_surfaces_interaction(engine, 'steel', 'steel', (0.1, 0.1, 0.1))

        stats = {}
        debug_on = True
        mu = SOLVER.get_friction_coefficient_vector(engine)
        J = SOLVER.compute_jacobian_matrix(engine, stats, debug_on)
        print('done test one')

    def test_two(self):
        print('running test two')

        def phi(x):
            return x[0]
        engine = API.Engine()
        V, T = VM.create_beam(2, 2, 2, 2.0, 1.0, 1.0)
        API.create_soft_body(engine, 'test', V, T)
        API.create_material(engine, 'steel')
        API.set_elasticity(engine, 'steel', 1.0, 0.3)
        API.set_viscosity(engine, 'steel', 1.0)
        API.set_mass_density(engine, 'steel', 1.0)
        API.set_constitutive_model(engine, 'steel', API.SVK)
        API.set_gravity(engine, 'test', (0, 0, -1))
        API.set_material(engine, 'test', 'steel')
        API.create_dirichlet_conditions(engine, 'test', phi)

        body = engine.bodies['test']
        vol = VM.compute_volumes(V, T)
        rho = 1.0
        A = SOLVER.ElementArrayUtil.assembly_csr(len(V), T, body.M_array)
        b = 100*np.ones((3*len(V,)))
        SOLVER.Native.apply_dirichlet_conditions(body.dirichlet_conditions, A, b)
        pyplot.spy(A)
        print('done test two')

    def test_three(self):
        print('running test three')
        V, T = VM.create_beam(2, 2, 2, 2.0, 1.0, 1.0)
        vol = VM.compute_volumes(V, T)
        rho = 1.0
        M_array = SOLVER.Native.compute_mass_element_array(rho, vol, T, is_lumped=False)
        print(M_array)
        M = SOLVER.ElementArrayUtil.assembly_csr(len(V), T, M_array)
        x = np.ones((3*len(V,)))
        z = SOLVER.ElementArrayUtil.prod_mat_vec(T, M_array, x)
        y = M.dot(x)
        print(vol)
        print(z)
        print(y)
        print('done test three')

    def test_traction_force(self):
        print('-- Running test of traction force -------------')

        def phi(x):
            return x[0]
        engine = API.Engine()
        V, T = VM.create_beam(2, 2, 2, 2.0, 1.0, 1.0)
        API.create_soft_body(engine, 'test', V, T)
        body = engine.bodies['test']
        print('vertices of body = ', body.x0)
        load = np.array([1., 0., 0.])
        API.create_traction_conditions(engine, 'test', phi, load)
        for tc in body.traction_conditions:
            print('triangle (', tc.i, tc.j, tc.k, ') with load = ', tc.traction)
        Ft = SOLVER.Native.compute_traction_forces(body.x, body.traction_conditions)
        print('Traction forces = ', Ft)
        print('-- Done traction force test -------------------')

    def test_forces(self):
        print('-- Running test of forces -------------')
        T = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
        V = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., -1.]])
        E, nu, rho = API.create_material_parameters()
        lambda_ = API.first_lame(E, nu)
        mu_ = API.first_lame(E, nu)
        vol0 = VM.compute_volumes(V, T)
        rho = 1.0
        x0 = np.copy(V)
        x = np.copy(x0)
        x[3, 2] += 0.1   # Pull top vertex slightly up (top tet is stretched a bit in z-direction)
        x[4, 2] -= 0.1   # Pull bottom vertex slightly down (bot tet is stretched a bit in z-direction)
        print('material coordinates:\n', x0)
        print('spatial coordinates:\n', x)
        g = np.array([0., 0., -10.])
        Fg = SOLVER.Native.compute_gravity_forces(len(V), T, vol0, rho, g)
        print('gravity:\n', Fg)
        Carray = SOLVER.Native.compute_damping_element_array(1.0, vol0, T, is_lumped=False)
        u = np.zeros_like(x0)
        u[:, 1] = 1   # Make all vertices move in y-direction with unit-speed
        print('velocities:\n', u)
        Fd = SOLVER.Native.compute_damping_forces(T, Carray, u)
        print('damping:\n', Fd)
        gradN = SOLVER.Native.compute_outward_face_vectors(x0, T)
        print('outward face vectors:', gradN.shape, '\n', gradN)
        print('face opposite m-node in first tet = ', gradN[0, :, 3])  # should point down in z-direction
        print('face opposite m-node in second tet = ', gradN[1, :, 3])  # should point up in z-direction
        D = SOLVER.Native.compute_D(x, T)
        print('Spatial edge matrices:\n', D)
        D0 = SOLVER.Native.compute_D(x0, T)
        print('Material edge matrices:\n', D0)
        invD0 = SOLVER.Native.compute_inverse_D0(x0, T)
        print('Inverse material edge matrices:\n', invD0)
        F = SOLVER.Native.compute_deformation_gradient(x, T, invD0)
        print('Deformation gradients:\n', F)
        Fe = SOLVER.Native.compute_elastic_forces(x, T, gradN, F, lambda_, mu_, API.SVK.pk1_stress)
        print('Elastic forces:\n', Fe)

    def test_solver(self):
        print('-- Running test of solver interface -------------')

        def wall_sdf(x):
            return x[0] + 2.9
        V, T = VM.create_beam(36, 6, 6, 6.0, 1.0, 1.0)
        engine = API.Engine()
        API.create_soft_body(engine, 'beam', V, T)
        API.set_velocity(engine, 'beam', (0, 0, 0))
        API.set_type(engine, 'beam', 'Free')
        API.create_dirichlet_conditions(engine, 'beam', wall_sdf)

        API.create_material(engine, 'steel')
        API.set_elasticity(engine, 'steel', 1.0, 0.3)
        API.set_viscosity(engine, 'steel', 1.0)
        API.set_mass_density(engine, 'steel', 1.0)
        API.set_constitutive_model(engine, 'steel', API.SVK)
        API.set_gravity(engine, 'beam', (0, 0, -10))
        API.set_material(engine, 'beam', 'steel')

        API.create_surfaces_interaction(engine, 'steel', 'steel', 0.5)

        x = SOLVER.get_position_vector(engine)
        u = SOLVER.get_velocity_vector(engine)
        data = {}
        Fext = SOLVER.compute_external_forces(engine, data, True)
        Fd = SOLVER.compute_damping_forces(u, engine, data, True)
        Fe = SOLVER.compute_elastic_forces(x, engine, data, True)
        Ft = SOLVER.compute_traction_forces(x, engine, data, True)
        print('x', x)
        print('u', u)
        print('Fext', Fext)
        print('Fd', Fd)
        print('Fe', Fe)
        print('Ft', Ft)
        data = SOLVER.stepper(0.01, engine, True)
        x = SOLVER.get_position_vector(engine)
        u = SOLVER.get_velocity_vector(engine)
        Fext = SOLVER.compute_external_forces(engine, data, True)
        Fd = SOLVER.compute_damping_forces(u, engine, data, True)
        Fe = SOLVER.compute_elastic_forces(x, engine, data, True)
        Ft = SOLVER.compute_traction_forces(x, engine, data, True)
        print('x', x)
        print('u', u)
        print('Fext', Fext)
        print('Fd', Fd)
        print('Fe', Fe)
        print('Ft', Ft)
        print('-- Done testing of solver interface -------------')


if __name__ == '__main__':
    #test = TestSoftBodySimulator()
    test = TestBlackBoxSimulator()
