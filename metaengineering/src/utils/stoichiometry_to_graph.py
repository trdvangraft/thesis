from collections import defaultdict
from typing import DefaultDict, List

import pandas as pd
import cobra
from cobra.util import create_stoichiometric_matrix

from more_itertools import flatten
import networkx as nx


def get_gene_reaction(model, enzymes):
    gene_to_reaction: DefaultDict[str, List[str]] = defaultdict(list)
    reaction_to_gene: DefaultDict[str, List[str]] = defaultdict(list)

    for gene in model.genes:
        if gene.id not in enzymes:
            continue

        for reaction in gene.reactions:
            gene_to_reaction[gene.id].append(reaction.id)
            reaction_to_gene[reaction.id].append(gene.id)
    return gene_to_reaction, reaction_to_gene

def is_precursor_metabolite(
    metabolite_model_id: str,
    valid_metabolites: List[str],
):
    for precursor_metabolite in valid_metabolites:
        if precursor_metabolite in metabolite_model_id:
            yield precursor_metabolite

def get_matching_precursor_metabolite(
    cobra_metabolite_id: str,
    valid_metabolites: List[str],
):
    for precursor_metabolite in valid_metabolites:
        if precursor_metabolite in cobra_metabolite_id:
            return precursor_metabolite

def get_edge_list(model, valid_enzymes, valid_metabolites):
    precursor_model_metabolites = list(filter(lambda x: any((pc_meta in x.id for pc_meta in valid_metabolites)), model.metabolites))
    precursor_model_metabolites_id = list(map(lambda x: x.id, precursor_model_metabolites))

    precursor_stiochiometric_df: pd.DataFrame = create_stoichiometric_matrix(model, array_type="DataFrame").loc[precursor_model_metabolites_id]
    precursor_stiochiometric_df = precursor_stiochiometric_df.loc[:, (precursor_stiochiometric_df != 0).any(axis=0)]

    _, reaction_to_gene = get_gene_reaction(model, valid_enzymes)

    precursor_reactions = list(map(lambda x: (reaction_to_gene[x], precursor_stiochiometric_df[x].values), precursor_stiochiometric_df.columns.to_list()))
    precursor_reactions = [(enzyme, stiochiometrie) for enzymes, stiochiometrie in precursor_reactions for enzyme in enzymes]

    edge_list_df = pd.DataFrame.from_records(precursor_reactions).T
    edge_list_df = edge_list_df.explode(edge_list_df.columns.to_list()).set_axis(edge_list_df.iloc[0], axis=1)
    edge_list_df = edge_list_df.drop(0).set_index(precursor_stiochiometric_df.index).groupby(edge_list_df.columns, axis=1).sum()
    edge_list_df['metabolite_id'] = [get_matching_precursor_metabolite(cobra_metabolite_id, valid_metabolites) for cobra_metabolite_id in edge_list_df.index]
    edge_list_df = edge_list_df.groupby('metabolite_id').sum()
    edge_list_df = edge_list_df.stack().rename_axis(['metabolite_id', 'enzyme']).rename("cardinality")
    edge_list_df = edge_list_df[edge_list_df != 0].reset_index()

    join_list = [
        {"nodes_to_join": ['2pg', '3pg'], "joined_name": "3pg;2pg"},
        {"nodes_to_join": ['g6p', 'g6p-B'], "joined_name": "g6p;g6p-B"},
    ]

    for join_item in join_list:
        edge_list_df = edge_list_df.apply(lambda x: x.replace({node: join_item['joined_name'] for node in join_item['nodes_to_join']}, regex=True))

    return edge_list_df

def build_graph(reaction_metabolite_enzyme):
    G = nx.Graph()
    for i in range(3):
        G.add_nodes_from(
            reaction_metabolite_enzyme.get_level_values(i).unique().to_list(),
            layer=i
        )

    r_to_m = [
        (metabolite_id, enzyme_id)
        for metabolite_id, enzyme_id in 
        zip(
            reaction_metabolite_enzyme.get_level_values('reaction_id'),    
            reaction_metabolite_enzyme.get_level_values('metabolite_id'),
        )
    ]

    r_to_e = [
        (metabolite_id, enzyme_id)
        for metabolite_id, enzyme_id in 
        zip(
            reaction_metabolite_enzyme.get_level_values('reaction_id'),
            reaction_metabolite_enzyme.get_level_values('enzyme_id')    
        )
    ]

    G.add_edges_from(r_to_m)
    G.add_edges_from(r_to_e)
    return G

def build_graph_directional(reaction_metabolite_enzyme_cardinality):
    G = nx.DiGraph()
    for i in range(3):
        G.add_nodes_from(
            reaction_metabolite_enzyme_cardinality.get_level_values(i).unique().to_list(),
            layer=i
        )

    # We connect the enzymes to reactions
    # bidirectional
    r_to_e = []
    for reaction_id, enzyme_id in zip(
        reaction_metabolite_enzyme_cardinality.get_level_values('reaction_id'),    
        reaction_metabolite_enzyme_cardinality.get_level_values('enzyme_id'),
    ):
        r_to_e.append((reaction_id, enzyme_id))
        r_to_e.append((enzyme_id, reaction_id))


    r_to_m = [
        (metabolite_id, reaction_id) if cardinality > 0 else (reaction_id, metabolite_id)
        for metabolite_id, reaction_id, cardinality in 
        zip(
            reaction_metabolite_enzyme_cardinality.get_level_values('metabolite_id'),
            reaction_metabolite_enzyme_cardinality.get_level_values('reaction_id'),
            reaction_metabolite_enzyme_cardinality.get_level_values('cardinality')
        )
    ]

    G.add_edges_from(r_to_m)
    G.add_edges_from(r_to_e)
    return G