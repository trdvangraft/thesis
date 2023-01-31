import tellurium as te
import roadrunner
# import antimony
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tempfile

from src.pipeline.dataloader import DataLoaderConfig
from src.pipeline.taskloader import TaskLoaderConfig
from src.settings.strategy import Strategy
from src.settings.tier import Tier

from src.pipeline.dataloader import DataLoader
from src.pipeline.taskloader import TaskFrame, TaskLoader

from typing import Callable
from tellurium.roadrunner import ExtendedRoadRunner

def search_variable_id(r: ExtendedRoadRunner, condition: Callable):
    return [variable_id for variable_id in r.getGlobalParameterIds() if condition(variable_id)]

def get_domain():
    domain = np.exp([-2, 2])
    return domain

def get_baseline_concentrations(r, metabolites):
    return np.array([r.getValue(metabolite_id) for metabolite_id in metabolites])

def single_mutation(r, orf_geneprefered_name_catalyzation_id, metabolites):
    dfs = []

    baseline_concentrations = get_baseline_concentrations()
    domain = get_domain()

    for catalyzation_id in orf_geneprefered_name_catalyzation_id['catalyzation_ids']:
        results = []
        print(catalyzation_id)
        for fc in np.arange(domain[0], domain[1], 0.1):
            r.setGlobalParameterByName(catalyzation_id, r.getGlobalParameterByName(catalyzation_id) * fc)
            result = r.simulate(0, 27000, 27001, selections=['time'] + metabolites)[-1, 1:]
            r.resetAll()
            results.append(np.log(baseline_concentrations) - np.log(result))

        dfs.append(
            pd.DataFrame(results, columns=metabolites) \
                .stack() \
                .reset_index() \
                # .assign(value=np.arange(domain[0], domain[1], 0.1)) \ 
                .assign(catalyzation_id=catalyzation_id) \
                
        )
    
    pd.concat(dfs).to_csv('./data/validation/single_mutation.csv')

def multi_mutation(r, orf_geneprefered_name_catalyzation_id, metabolites):
    dfs = []
    baseline_concentrations = get_baseline_concentrations(r, metabolites)
    domain = get_domain()
    column_name = metabolites.copy()
    column_name.extend(["sample_id", "catalyzation_change"])
    print(column_name)

    results = []
    for sample_id in range(100):
        r.resetAll()
        sample = (domain[1] - domain[0]) * np.random.random(size=12) + domain[0]
        for catalyzation_id, sample_value in zip(orf_geneprefered_name_catalyzation_id['catalyzation_ids'], sample):
            r.setGlobalParameterByName(catalyzation_id, r.getGlobalParameterByName(catalyzation_id) * sample_value)
        try:
            result = r.simulate(0, 27000, 27001, selections=['time'] + metabolites)[-1, 1:]
            configuration = {catalyzation_id: [sample_value] for catalyzation_id, sample_value in zip(orf_geneprefered_name_catalyzation_id['catalyzation_ids'], sample)}
            t = pd.DataFrame(
                [np.log(baseline_concentrations) - np.log(result)],
                columns=metabolites
            ).stack().reset_index().assign(sample_id=sample_id).rename({'level_1': 'metabolite_id', 0: 'metabolite_concentration'}, axis=1)
            tt = pd.DataFrame.from_dict(configuration).stack().reset_index().assign(sample_id=sample_id).rename({'level_1': 'enzyme_id', 0: 'enzyme_concentration'}, axis=1)
            ttt = t.merge(tt, on='sample_id')[['sample_id', 'enzyme_id', 'metabolite_id', 'enzyme_concentration', 'metabolite_concentration']]
            results.append(ttt)
        except RuntimeError:
            pass
        finally:
            r.resetAll()

    dfs.append(
        pd.concat(results, ignore_index=True)
    )
    
    pd.concat(dfs, ignore_index=True).to_json("./data/validation/multi_mutation.json")

def main():
    with open('./experiments/paper_validation/kinetic_model.txt', 'r') as file:
        model = file.read()

    protein_info = pd.read_json("/home/tvangraft/tudelft/thesis/metaengineering/data/training/gene_annotation.json") \
        .T.explode('gene_prefered_name').reset_index(names=['orf'])

    r = te.loada(model)
    metabolites = ['F6P', 'PEP', 'PYR', 'P3G', 'P2G']

    enzyme_catalyzation_ids = search_variable_id(r, lambda x: "cat" in x)
    enzymes = [enzyme_id.split("_")[1] for enzyme_id in enzyme_catalyzation_ids]

    _df = pd.DataFrame(data={
        'gene_prefered_name': enzymes,
        'catalyzation_ids': enzyme_catalyzation_ids,
    })

    orf_geneprefered_name_catalyzation_id = protein_info.merge(
        _df, left_on='gene_prefered_name', right_on='gene_prefered_name'
    )[['orf', 'gene_prefered_name', 'catalyzation_ids']]

    r = te.loada(model)
    # single_mutation(r, orf_geneprefered_name_catalyzation_id, metabolites)
    multi_mutation(r, orf_geneprefered_name_catalyzation_id, metabolites)

    

    dfs = []


if __name__ == '__main__':
    main()