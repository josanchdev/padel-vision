"""Array types shared across the project, so each is defined once."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

ImageArray = npt.NDArray[np.uint8]
"""A BGR frame as OpenCV reads it: (height, width, 3)."""
FloatArray = npt.NDArray[np.float32]
