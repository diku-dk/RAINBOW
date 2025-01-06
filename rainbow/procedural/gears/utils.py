
import math


def involute(alpha: float):
    return math.tan(alpha) - alpha


def roll_angle(r_base: float, r_top: float) -> float:
    return math.sqrt((r_top / r_base) ** 2 - 1)
