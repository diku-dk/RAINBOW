"""Shared array and state type aliases for the soft-body package."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

Array: TypeAlias = np.ndarray
Vertices: TypeAlias = np.ndarray
Elements: TypeAlias = np.ndarray
Forces: TypeAlias = np.ndarray

__all__ = ["Array", "Elements", "Forces", "Vertices"]
