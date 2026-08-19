"""Shared Pydantic schemas for Planner backend.

This module provides all data schemas used across the application,
organized by domain:
- intent: User intent and conversation schemas
- specification: Traffic profile and SLO target schemas
- recommendation: GPU config, scores, and recommendation schemas
"""

from enum import StrEnum

from .intent import ConversationMessage, DeploymentIntent, GpuPreference
from .recommendation import (
    ConfigurationScores,
    DeploymentBundle,
    DeploymentConfiguration,
    DeploymentRecommendation,
    GPUConfig,
    RankedRecommendations,
)
from .specification import (
    DeploymentSpecification,
    Priorities,
    PriorityEntry,
    QualityWeights,
    SLORange,
    SLOTargets,
    TrafficProfile,
    WorkloadProfile,
)


class DeploymentMode(StrEnum):
    """Valid deployment modes."""

    PRODUCTION = "production"
    SIMULATOR = "simulator"
