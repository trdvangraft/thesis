from enum import Enum


class Tasks(Enum):
    SINGLE_METABOLITE = 'single_metabolite_task'
    ALL_METABOLITE = 'all_metabolite_task'


class DataOrientation(Enum):
    SIMPLE = 'genotype'
    LIST = 'genotype_protein'
    FULL = 'genotype_protein_metabolite'
