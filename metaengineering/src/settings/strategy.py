from enum import Enum


class Strategy(Enum):
    METABOLITE_CENTRIC = "model_per_metabolite"
    GENOTYPE_CENTRIC = "model_per_genotype"
    MODEL_CENTRIC = "model_per_metabolite_per_genotype"
    ALL = "full_dataset"