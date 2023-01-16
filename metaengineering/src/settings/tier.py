from enum import Enum



class Tier(Enum):
    TIER0 = "baseline_dataset"
    TIER1 = "ppi_dataset"
    TIER2 = "protein_stoichiometry"
    TIER3 = "protein_metabolite_stoichiometry"

    @classmethod
    def get_order(cls, tier):
        if tier == Tier.TIER0:
            return 0
        elif tier == Tier.TIER1:
            return 1
        elif tier == Tier.TIER2:
            return 2
        elif tier == Tier.TIER3:
            return 3