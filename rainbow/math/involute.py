from __future__ import annotations
from typing import Callable

import numpy as np


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


def create_involute_x_function(r_base: float, offset: float) -> Callable[[float], float]:
    """
    Create a function that computes the x-coordinate of the involute curve.
    
    :param r_base:  The radius of the base circle.
    :param offset:  The offset angle.
    :return:        A function that computes the x-coordinate of the involute curve at a given angle.
    """
    
    def f(alpha: float) -> float:
        return r_base * (np.cos(alpha + offset) + alpha * np.sin(alpha + offset))
    return f


def create_involute_y_function(r_base: float, offset: float) -> Callable[[float], float]:
    """
    Create a function that computes the y-coordinate of the involute curve.
    
    :param r_base:  The radius of the base circle.
    :param offset:  The offset angle.
    :return:        A function that computes the y-coordinate of the involute curve at a given angle.
    """
    
    def f(alpha: float) -> float:
        return r_base * (np.sin(alpha + offset) - alpha * np.cos(alpha + offset))
    return f
