from dataclasses import dataclass, field
from typing import List, Callable

from src.settings.tier import Tier


@dataclass
class DataLoaderConfig:
    additional_frames: List[str] = field(default_factory=list)
    additional_transforms: List[str] = field(default_factory=list)
    additional_filters: List[str] = field(default_factory=list)

@dataclass
class ParsedDataLoaderConfig:
    additional_frames: List[Callable] = field(default_factory=list)
    additional_transforms: List[Callable] = field(default_factory=list)
    additional_filters: List[Callable] = field(default_factory=list)

@dataclass
class TaskLoaderConfig:
    data_throttle: int = 1.0
    tier: Tier = Tier.TIER0