from src.pipeline.dataloader import DataLoader
from more_itertools import flatten

DataLoader.DATA_FOLDER = './data/training/'
dl = DataLoader()
protein_metabolite_df = dl.get_simple_protein_metabolite_dataframe()
ENZYMES = protein_metabolite_df.to_df().columns.to_list()

METABOLITES = protein_metabolite_df.obs.columns.to_list()
PRECURSOR_METABOLITES = [
    'g6p;g6p-B', 'f6p', 'dhap', '3pg;2pg',
    'pep', 'pyr', 'r5p', 'e4p', 'accoa', 'akg', 'oaa',
]

PRECURSOR_METABOLITES_NO_TRANSFORM = [
    'f6p', 'dhap'
    'pep', 'pyr', 'r5p', 'e4p', 'accoa', 'akg', 'oaa',
]