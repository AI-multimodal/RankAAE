#!/usr/bin/env python3


from pymatgen.analysis.chemenv.coordination_environments\
    .chemenv_strategies import MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments\
    .structure_environments import LightStructureEnvironments as LSE


def get_cesym(lgf, structure, sites, maximum_distance_factor=2.0):
    # TODO: docstring

    # doc: http://pymatgen.org/_modules/pymatgen/analysis/chemenv/
    #      coordination_environments/coordination_geometry_finder.html
    lgf.setup_structure(structure)

    # doc: http://pymatgen.org/_modules/pymatgen/analysis/chemenv/
    #      coordination_environments/
    #      chemenv_strategies.html#MultiWeightsChemenvStrategy.
    #      stats_article_weights_parameters
    strategy = MultiWeightsChemenvStrategy.stats_article_weights_parameters()

    # returns all information about the structure; se is a structure object
    mdf = maximum_distance_factor
    se = lgf.compute_structure_environments(
        maximum_distance_factor=mdf, only_cations=False, only_indices=sites
    )

    lse = LSE.from_structure_environments(
        strategy=strategy, structure_environments=se
    )

    coor = lse.coordination_environments

    # ce = chemical environment
    # csm = continuous symmetry measure
    # low csm = closer to that predicted geometry
    try:
        return [[loc[0]['ce_symbol'] for loc in coor if loc is not None],
                [loc[0]['csm'] for loc in coor if loc is not None]]
    except IndexError:  # list out of range
        return ['index_error', 'index_error']
