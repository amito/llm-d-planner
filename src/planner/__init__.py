"""Backend source package for Planner."""

try:
    from planner._version import __version__
except ModuleNotFoundError:
    __version__ = "0.0.dev0"

from planner.config import PlannerConfig
from planner.errors import PlannerError
from planner.planner import Planner
from planner.shared.schemas import (
    DeploymentBundle,
    DeploymentConfiguration,
    DeploymentIntent,
    DeploymentSpecification,
    GPUConfig,
    GpuPreference,
    RankedRecommendations,
    SLOTargets,
    WorkloadProfile,
)

__all__ = [
    "__version__",
    "Planner",
    "PlannerConfig",
    "PlannerError",
    "DeploymentConfiguration",
    "DeploymentIntent",
    "DeploymentSpecification",
    "DeploymentBundle",
    "GPUConfig",
    "GpuPreference",
    "RankedRecommendations",
    "SLOTargets",
    "WorkloadProfile",
]
