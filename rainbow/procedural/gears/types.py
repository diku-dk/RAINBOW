
import numpy as np

import rainbow.math.involute as INV

class InvoluteGearSpec:
    def __init__(self, m: float, z: int, alpha: float = 20.0) -> None:
        """
        Create a new involute gear specification.
        
        :param m:       The module of the gear.
        :param z:       The number of teeth.
        :param alpha:   The pressure angle in degrees.
        """
        self.m = m # module
        if not (m > 0):
            raise ValueError("Module m must be positive.")
        self.z = z # number of teeth
        if not (z > 4):
            raise ValueError("Number of teeth z must be greater than 4.")
        if not (0 <= alpha <= 90):
            raise ValueError("Pressure angle alpha must be between 0 and 90 degrees.")
        self.alpha = np.deg2rad(alpha) # pressure angle
        
        self.ha = self.m # addendum
        self.hd = 1.25 * self.m # dedendum
        
        self.rp = self.m * self.z / 2 # pitch circle radius
        self.ra = self.rp + self.ha # addendum circle radius
        self.rd = self.rp - self.hd # dedendum circle radius
        self.rb = np.cos(self.alpha) * self.rp # base circle radius
        
        self.t_min = 0 # minimum roll angle
        if self.rb < self.rd:
            self.t_min = INV.roll_angle(self.rb, self.rd)
        self.t_max = INV.roll_angle(self.rb, self.ra) # maximum roll angle

        self.delta_bp = INV.involute(self.alpha) # involute pitch from base to pitch circle
        self.delta_ba = INV.involute(np.arctan(self.t_max)) # involute pitch from base to addendum circle
        self.delta_pa = self.delta_ba - self.delta_bp # involute pitch from pitch to addendum circle
        
        self.gamma_p = np.pi / self.z # angular pitch
        self.gamma_b = self.gamma_p + 2 * self.delta_bp # base pitch
        self.gamma_a = self.gamma_p - 2 * self.delta_pa # addendum pitch
        
        self.theta_i = np.linspace(0, 2 * np.pi, z + 1) # tooth angles
