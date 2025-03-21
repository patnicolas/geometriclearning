
__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2023, 2025  All rights reserved."

import numpy as np
from typing import AnyStr
from dataclasses import dataclass

"""
    Wrapper for Point or Matrix on SO3 manifold that leverages the Geomstats library.
    @param group_element Point on the Manifold
    @param identity_element  Base or reference point (i.g., Identity matrix)
    @param descriptor Description of the point
"""


@dataclass
class LieElement:
    group_element: np.array     # Point on the Manifold
    identity_element: np.array  # Reference point (Identity by default)
    descriptor: AnyStr = 'Lie Element'
