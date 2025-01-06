
import math
import numpy as np

from .utils import roll_angle

class InvoluteGearSpec:
    def __init__(self, module: float, teeth: int, pressure_angle: float = 20.0) -> None:
        self.module = module
        self.teeth = teeth
        self.pressure_angle = pressure_angle
        self.pressure_angle_rad = np.pi * self.pressure_angle / 180.0
        
        self.addendum = self.module
        self.dedendum = 1.25 * self.module
        self.tooth_depth = 2.25 * self.module
        self.working_depth = 2.0 * self.module
        self.root_clearance = 0.25 * self.module
        self.dedendum_fillet_radius = 0.38 * self.module
        self.pitch = np.pi * self.module
        self.tooth_thickness = self.pitch / 2.0
        
        self.reference_radius = self.module * self.teeth / 2.0
        self.dedendum_radius = self.reference_radius - self.dedendum
        self.addendum_radius = self.reference_radius + self.addendum
        self.base_radius = self.reference_radius * math.cos(self.pressure_angle_rad)
        
        self.pitch_roll = roll_angle(self.base_radius, self.reference_radius)
        self.top_roll = roll_angle(self.base_radius, self.addendum_radius)
        
        
