from enum import Enum


class Tasks(Enum):
    SINGLE_METABOLITE = 'single_metabolite_task'
    ALL_METABOLITE = 'all_metabolite_task'


class DataOrientation(Enum):
    SIMPLE = 'genotype'
    LIST = 'genotype_protein'
    FULL = 'genotype_protein_metabolite'


class Strategy(Enum):
    METABOLITE_CENTRIC = "model_per_metabolite"
    GENOTYPE_CENTRIC = "model_per_genotype"
    MODEL_CENTRIC = "model_per_metabolite_per_genotype"
    ALL = "full_dataset"
