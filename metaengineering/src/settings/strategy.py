from enum import Enum


class Strategy(Enum):
    METABOLITE_CENTRIC = "model_per_metabolite"
    GENOTYPE_CENTRIC = "model_per_genotype"
    MODEL_CENTRIC = "model_per_metabolite_per_genotype"
    ALL = "full_dataset"
    ONE_VS_ALL = "one_vs_all"

    @staticmethod
    def from_str(label):
        if label == 'Strategy.METABOLITE_CENTRIC':
            return Strategy.METABOLITE_CENTRIC
        elif label == 'Strategy.ONE_VS_ALL':
            return Strategy.ONE_VS_ALL
        elif label == 'Strategy.ALL':
            return Strategy.ALL
    
    @classmethod
    def get_order(cls, strategy):
        if strategy == Strategy.ALL:
            return 0
        elif strategy == Strategy.METABOLITE_CENTRIC:
            return 1
        elif strategy == Strategy.ONE_VS_ALL:
            return 2
    
    def __le__(self, b):
        return Strategy.get_order(self) <= Strategy.get_order(b)
    
    def __lt__(self, b):
        return Strategy.get_order(self) < Strategy.get_order(b)