from enum import Enum



class Tier(Enum):
    TIER0 = "baseline_dataset"
    TIER1 = "ppi_dataset"
    TIER2 = "protein_stoichiometry"
    TIER3 = "protein_metabolite_stoichiometry"

    @staticmethod
    def from_str(label):
        if label == 'Tier.TIER0':
            return Tier.TIER0
        elif label == 'Tier.TIER1':
            return Tier.TIER1
        elif label == 'Tier.TIER2':
            return Tier.TIER2
        elif label == 'Tier.TIER3':
            return Tier.TIER3

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
    
    def __le__(self, b):
        return Tier.get_order(self) <= Tier.get_order(b)
    
    def __lt__(self, b):
        return Tier.get_order(self) < Tier.get_order(b)