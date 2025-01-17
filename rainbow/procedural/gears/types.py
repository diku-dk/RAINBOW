
import numpy as np

import rainbow.math.involute as INV

class GearSpec:
    """A class representing the specification of an involute gear.
    
    The gear specification includes the module, number of teeth, pressure angle, helix angle, and internal gear indicator.
    """
    def __init__(self, m: float, z: int, pressure_angle: float = 20.0, helix_angle: float | None = None, is_internal: bool = False) -> None:
        """
        Create a new involute gear specification.
        
        :param m: Module of the gear.
        :param z: Number of teeth of the gear.
        :param pressure_angle: Pressure angle of the gear in degrees.
        :param helix_angle: Helix angle of the gear in degrees. If None, the gear is spur.
        :param is_internal: Internal gear indicator.
        
        :raises ValueError: If m is not positive, z is less than or equal to 4, or pressure angle is not between 0 and 90 degrees.
        """
        if not (m > 0):
            raise ValueError("Module m must be positive.")
        if not (z > 4):
            raise ValueError("Number of teeth z must be greater than 4.")
        if not (0 <= pressure_angle <= 90):
            raise ValueError("Pressure angle alpha must be between 0 and 90 degrees.")
        
        self.m = m # module
        self.z = z # number of teeth
        self.alpha = np.deg2rad(pressure_angle) # pressure angle
        self.beta = np.deg2rad(helix_angle) if helix_angle is not None else None # helix angle
        self.is_internal = is_internal # internal or external gear indicator
        
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
        self.delta_bd = None if self.rb >= self.rd else INV.involute(np.arctan(self.t_min)) # involute pitch from base to dedendum circle (Only applies if rb < rd)
        
        self.gamma_p = np.pi / self.z # angular pitch
        self.gamma_b = self.gamma_p + 2 * self.delta_bp # base pitch
        self.gamma_a = self.gamma_p - 2 * self.delta_pa # addendum pitch
        self.gamma_d = self.gamma_b if self.delta_bd is None else self.gamma_b - 2 * self.delta_bd # dedendum pitch
        
        self.theta_i = np.array([i * 2 * self.gamma_p for i in range(z)]) # tooth angles


class Gear:
    def __init__(self, spec: GearSpec, V: np.ndarray, T: np.ndarray) -> None:     
        """Create a new involute gear.
        
        :param spec: Gear specification.
        :param V: Vertex array of the gear.
        :param T: Triangle array of the gear.
        """   
        self.spec = spec
        self.V = V
        self.T = T
