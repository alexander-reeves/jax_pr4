"""
JAX-ified Planck PR4 CMB likelihoods

This package provides JAX-compatible implementations of Planck PR4 likelihoods:
- CamSpec (high-ell TT, TE, EE)
- Hillipop (high-ell TT, TE, EE) 
- Lollipop (low-ell EE, BB, EB)

All likelihoods are designed to work with direct Cl inputs rather than cosmological parameters.
"""

from . import module
from . import utils
from . import foregrounds_hillipop

# Import likelihood classes
try:
    from .camspec_pr4 import CamSpecPR4
except ImportError:
    CamSpecPR4 = None

try:
    from .hillipop import HillipopPR4
except ImportError:
    HillipopPR4 = None

try:
    from .lollipop import LollipopPR4
except ImportError:
    LollipopPR4 = None

__version__ = "0.1.0"
__all__ = ["module", "utils", "foregrounds_hillipop", "CamSpecPR4", "HillipopPR4", "LollipopPR4"] 