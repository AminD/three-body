"""Three-body problem: simulation, integrator comparison and chaos."""

from .integrators import integrate, integrate_adaptive
from .scenarios import SCENARIOS, chaotic, figure_eight, lagrange, load_config
from .state import InitialConditions, Trajectory

__all__ = [
    "InitialConditions",
    "Trajectory",
    "integrate",
    "integrate_adaptive",
    "SCENARIOS",
    "figure_eight",
    "lagrange",
    "chaotic",
    "load_config",
]
__version__ = "0.1.0"
