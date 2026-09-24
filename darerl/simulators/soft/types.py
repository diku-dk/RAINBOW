"""Shared public types for the soft-body package."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

from .material import Material
from .mesh import TetMesh

Array: TypeAlias = np.ndarray

__all__ = ["Array", "Material", "TetMesh"]
