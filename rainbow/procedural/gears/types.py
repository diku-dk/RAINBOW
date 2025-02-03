
import numpy as np

import rainbow.math.involute as INV
import rainbow.math.quaternion as Q
import rainbow.geometry.surface_mesh as MESH

class GearSpec:
    """A class representing the specification of an involute gear.
    
    The gear specification includes the module, number of teeth, pressure angle, helix angle, and internal gear indicator.
    """
    def __init__(
        self, 
        m: float, 
        z: int, 
        pressure_angle: float = 20.0, 
        helix_angle: float | None = None, 
        is_internal: bool = False, 
        bevel_cone_angle: float | None = None
    ) -> None:
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
        
        self.is_bevel = bevel_cone_angle is not None # bevel gear indicator
        self.bevel_cone_angle = np.deg2rad(bevel_cone_angle) if self.is_bevel else None # bevel cone angle
        
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
        self.delta_ba = INV.involute(np.cos(self.rb / self.ra)) # involute pitch from base to addendum circle
        self.delta_pa = self.delta_ba - self.delta_bp # involute pitch from pitch to addendum circle
        self.delta_bd = None if self.rb >= self.rd else INV.involute(np.cos(self.rb / self.rd)) # involute pitch from base to dedendum circle (Only applies if rb < rd)
        
        self.gamma_p = np.pi / self.z # angular pitch
        self.gamma_b = self.gamma_p + 2 * self.delta_bp # base pitch
        self.gamma_a = self.gamma_p - 2 * self.delta_pa # addendum pitch
        self.gamma_d = self.gamma_b if self.delta_bd is None else self.gamma_b - 2 * self.delta_bd # dedendum pitch
        
        self.theta_i = np.array([i * 2 * self.gamma_p for i in range(z)]) # tooth angles


class Gear:
    def __init__(self, spec: GearSpec, mesh: MESH.Mesh, position: np.ndarray | None = None, orientation: np.ndarray | None = None) -> None:     
        """Create a new involute gear.
        
        :param spec: Gear specification.
        :param V: Vertex array of the gear.
        :param T: Triangle array of the gear.
        """
        self.spec = spec
        self.position = position if position is not None else np.zeros(3)
        self.orientation = orientation if orientation is not None else Q.identity()
        self.mesh = mesh


class PlanetaryGearSpec:
    def __init__(self, m: float, z_sun: int, z_planet: int, z_ring: int, N_planet: int = 3, pressure_angle: float = 20, helix_angle: float | None = None) -> None:
        self.m = m
        self.z_sun = z_sun
        self.z_planet = z_planet
        self.z_ring = z_ring
        
        self.N_planet = N_planet
        
        if not (self.z_ring == self.z_sun + 2 * self.z_planet):
            raise ValueError("The number of teeth in the ring gear must be equal to the sum of the sun and twice the planet gear teeth.")
        
        if not ((self.z_sun + self.z_ring) / self.N_planet).is_integer():
            raise ValueError("The number of planet gears must be a divisor of the sum of the sun and ring gear teeth.")
        
        if not (self.z_planet + 2 < (self.z_sun + self.z_ring) * np.sin(np.pi / self.N_planet)):
            raise ValueError("Planet gears have interference.")
        
        self.alpha = pressure_angle
        self.beta = helix_angle
        
        self.planet_angles = np.linspace(0, 2 * np.pi, self.N_planet, endpoint=False)
        
        self.sun_spec = GearSpec(m, z_sun, pressure_angle, helix_angle)
        self.planet_spec = GearSpec(m, z_planet, pressure_angle, -helix_angle if helix_angle is not None else None)
        self.ring_spec = GearSpec(m, self.z_ring, pressure_angle, -helix_angle if helix_angle is not None else None, is_internal=True)


class PlanetaryGear:
    def __init__(self,
            planetary_spec: PlanetaryGearSpec,
            sun_gear: Gear,
            planet_gears: list[Gear],
            ring_gear: Gear) -> None:
        self.planetary_spec = planetary_spec
        
        self.sun_gear = sun_gear
        self.planet_gears = planet_gears
        self.ring_gear = ring_gear
        
        self.gears = [self.sun_gear] + self.planet_gears + [self.ring_gear]


class MatingSpec:
    def __init__(self, driving_gear: GearSpec, driven_gear: GearSpec, position: np.ndarray, rotation: np.ndarray) -> None:
        self.driving_gear = driving_gear
        self.driven_gear = driven_gear
        self.position = position
        self.rotation = rotation
