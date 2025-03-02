
import rainbow.simulators.prox_rigid_bodies.types as TYPES

from .USD import USD


class UsdManager:
    def __init__(self, file_path: str):
        self.usd_scene = USD(file_path)
        self.last_step = 0

    def initialize(self, engine: TYPES.Engine) -> None:
        self.usd_scene.set_frames_per_second(engine.params.time_step)
        for body in engine.bodies.values():
            self.usd_scene.add_rigid_body(body.name, body.shape.mesh.V, body.shape.mesh.T)
            self.usd_scene.update_rigid_body(body.name, body.r, body.q, 0.0)

    def save_step(self, step: int, engine: TYPES.Engine) -> None:
        for body in engine.bodies.values():
            self.usd_scene.update_rigid_body(body.name, body.r, body.q, step)
        self.last_step = max(self.last_step, step)

    def save(self) -> None:
        self.usd_scene.set_animation_time(self.last_step)
        while True:
            try:
                self.usd_scene.save()
                break
            except Exception as e:
                continue
