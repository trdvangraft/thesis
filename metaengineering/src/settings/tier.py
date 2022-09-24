from enum import Enum


class Tier(Enum):
    TIER0 = "baseline_dataset"
    TIER1 = "simple_dataset"
    TIER2 = "feature_rich_dataset"
    TIER3 = "interaction_dataset"
    TIER4 = "graph_dataset"