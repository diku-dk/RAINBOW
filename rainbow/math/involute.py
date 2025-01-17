from __future__ import annotations
from typing import Callable

import numpy as np


class InvoluteCurve:
    def __init__(self, r_base: float, offset: float):
        self.rb = r_base
        self.a = offset
    
    def x(self, t: float) -> float:
        return self.rb * (np.cos(t + self.a) + t * np.sin(t + self.a))
    
    def y(self, t: float) -> float:
        return self.rb * (np.sin(t + self.a) - t * np.cos(t + self.a))
    
    def __call__(self, t: float) -> np.ndarray:
        return np.vstack((self.x(t), self.y(t))).T


def involute(alpha: float) -> float:
    """
    Compute the involute of a circle.

    :param alpha:   The angle of the involute.
    :return:        The involute of the circle at the given angle.
    """
    return np.tan(alpha) - alpha


def roll_angle(r_base: float, r: float) -> float:
    """
    Compute the roll angle.

    :param r_base:  The radius of the inner circle.
    :param r_top:   The radius of the outer circle.
    :return:        The roll angle of the inner circle that will create an involute curve connecting to the
                    outer circle.
    """
    
    return np.sqrt((r / r_base) ** 2 - 1)
