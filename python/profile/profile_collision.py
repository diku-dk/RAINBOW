from cProfile import label
from distutils.log import debug
import enum
from pstats import Stats
from re import I
from selectors import SelectorKey
import sys
import os
from turtle import update
import numpy as np
from time import time
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import isl.simulators.prox_rigid_bodies.api as API
import isl.simulators.prox_rigid_bodies.solver as SOLVER
import isl.simulators.prox_rigid_bodies.collision_detection as CD
import isl.geometry.surface_mesh as MESH
import isl.math.vector3 as V3
import isl.math.quaternion as Q
from isl.util.timer import Timer


class Profile:
    def __init__(self, funcs, inputs):
        self.funcs = funcs
        self.inputs = inputs
        self.outputs = []
        self.simple_profiles = []

    def run(self):
        # for e,_,_ in self.inputs:
        #     print(f"Contact points before: {e.contact_points}")
        for i,f in enumerate(self.funcs):
            timer = Timer("Stepper")
            timer.start()
            exec = f(*self.inputs[i])
            timer.end()
            self.outputs.append(exec)
            self.simple_profiles.append(timer.elapsed)
        
        p_e,_,_ = self.inputs[0]
        # for e,_,_ in self.inputs:
        #     print(f"Contact points after: {len(e.contact_points)}")
        for c_e,_,_ in self.inputs[1:]:
            
            if not p_e.contact_points == c_e.contact_points:
                print(f'Number of contact points old: {len(p_e.contact_points)}')
                print(f'Number of contact points new: {len(c_e.contact_points)}')
                # for i, pc in enumerate(p_e.contact_points):
                #     cc = c_e.contact_points[i]
                #     if not pc == cc:
                #         print(f"{pc} ==\n {cc}: {pc == cc}\n")
        


class ProfileCollisionDetection(Profile):
    def __init__(self, funcs, input):
        Profile.__init__(self, funcs, input)
        self.stats_merge = {}

    def mergeStats(self):
        outputs = self.outputs
        parent = outputs[0]
        for k in parent.keys():
            self.stats_merge[k] = [[parent[k]]]
            for c in outputs[1:]:
                (self.stats_merge[k])[0].append(c[k])

    def __add__(self, o):
        if self.stats_merge == {}:
            self.mergeStats()
        addition = {}
        for k in self.stats_merge.keys():
            addition[k] = self.stats_merge[k] + o.stats_merge[k]
        addProfile = ProfileCollisionDetection([], None)
        addProfile.stats_merge = addition
        return addProfile


class ProfileViewerSimple:
    def __init__(self, profiles, labels, title):
        self.profiles = profiles
        self.labels = labels
        self.title = title

    def show(self):
        _, ax = plt.subplots()
        colors = np.random.rand(len(self.labels), 3)
        for i, ps in enumerate(self.profiles):
            for j, p in enumerate(ps.simple_profiles):
                if i <= 0:
                    ax.scatter(i, p, label=self.labels[j], color=colors[j])
                else:
                    ax.scatter(i, p, color=colors[j])
        plt.title(self.title)
        plt.legend()
        plt.show()

    def show_detail(self):
        parent = self.profiles[0]
        for c in self.profiles[1:]:
            parent += c

        for k in parent.stats_merge.keys():
            _, ax = plt.subplots()
            prof_func = parent.stats_merge[k]
            colors = np.random.rand(len(self.labels), 3)
            for i, r in enumerate(prof_func):
                for j, col in enumerate(r):
                    if i <= 0:
                        ax.scatter(i, col, label=self.labels[j], color=colors[j])
                    else:
                        ax.scatter(i, col, color=colors[j])
            plt.grid(True)
            plt.title(k)
            plt.legend()
            plt.show()


def initCollisionDetectionInput(number, envelope, resolution, sphere_resolution):
    output = []
    for _ in range(number):
        engine = API.Engine()
        engine.params.envelope = envelope
        engine.params.resolution = resolution
        API.create_rigid_body(engine, "parent_bone")
        API.create_rigid_body(engine, "child_bone")
        V, T = MESH.create_sphere(2.0, sphere_resolution, sphere_resolution)
        mesh = API.create_mesh(V, T)
        API.create_shape(engine, "parent_shape", mesh)
        V, T = MESH.create_sphere(2.0, sphere_resolution, sphere_resolution)
        mesh = API.create_mesh(V, T)
        API.create_shape(engine, "child_shape", mesh)
        API.connect_shape(engine, "parent_bone", "parent_shape")
        API.connect_shape(engine, "child_bone", "child_shape")
        API.set_mass_properties(engine, "parent_bone", 1.0)
        API.set_mass_properties(engine, "child_bone", 1.0)
        API.set_orientation(engine, "parent_bone", Q.identity(), use_model_frame=True)
        API.set_position(
            engine, "parent_bone", V3.make(2.0, 0.0, 0.0), use_model_frame=True
        )
        API.set_orientation(engine, "child_bone", Q.identity(), use_model_frame=True)
        API.set_position(
            engine, "child_bone", V3.make(-2.0, 0.0, 0.0), use_model_frame=True
        )
        API.set_body_type(engine, "parent_bone", "fixed")
        output.append((engine, {}, True))
    return output

def dummyCollsion(input1, input2, input3):
    return {"update_bvh" : 0,
            "narrow_phase" : 0,
            "number_of_overlaps" : 0,
            "model_space_update" : 0,
            "contact_optimization": 0,
            "contact_point_generation": 0,
            "contact_determination": 0,
            "contact_point_reduction": 0,
            "collision_detection_time": 0}

def main():

    func_list = [CD.run_collision_detection, CD.run_collision_detection_faster]
    number_of_functions = len(func_list)

    inputs = initCollisionDetectionInput(number_of_functions, 3.5, 64, 4)
    Profiler1 = ProfileCollisionDetection(
        func_list, inputs
    )
    inputs = initCollisionDetectionInput(number_of_functions, 3.5, 64, 8)
    Profiler2 = ProfileCollisionDetection(
        func_list, inputs
    )
    inputs = initCollisionDetectionInput(number_of_functions, 3.5, 64, 16)
    Profiler3 = ProfileCollisionDetection(
        func_list, inputs
    )
    inputs = initCollisionDetectionInput(number_of_functions, 3.5, 64, 24)
    Profiler4 = ProfileCollisionDetection(
        func_list, inputs
    )
    # inputs = initCollisionDetectionInput(number_of_functions, 3.5, 64, 32)
    # Profiler5 = ProfileCollisionDetection(
    #     [CD.run_collision_detection, CD.run_collision_detection_faster], inputs
    # )
    # inputs = initCollisionDetectionInput(number_of_functions, 3.5, 64, 64)
    # Profiler6 = ProfileCollisionDetection(
    #     [CD.run_collision_detection, CD.run_collision_detection_faster], inputs
    # )
    Profiler1.run()
    Profiler1.mergeStats()

    Profiler2.run()
    Profiler2.mergeStats()

    Profiler3.run()
    Profiler3.mergeStats()

    Profiler4.run()
    Profiler4.mergeStats()

    # Profiler5.run()
    # Profiler5.mergeStats()

    # Profiler6.run()
    # Profiler6.mergeStats()
    

    ProfileViewerSimple(
        [Profiler1, Profiler2, Profiler3, Profiler4], ["Collision Detection old", "Collision Detection new"], "Test plot"
    ).show_detail()


if __name__ == "__main__":
    main()
