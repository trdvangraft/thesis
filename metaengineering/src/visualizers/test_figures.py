import pandas as pd

import seaborn as sns
from seaborn import FacetGrid

class TestFigures():
    def __init__(self, test_df: pd.DataFrame) -> None:
        self.test_df = test_df
    
    def r2_per_metabolite(self):
        _df = self.test_df.sort_values(['pathway', 'metabolite_id'])
        g = sns.barplot(
            data=_df,
            x='metabolite_id',
            y='r2',
            hue='pathway',
            dodge=False,
            palette='deep',
        )
        # g.set(yscale='symlog')
        g.set_xticklabels(g.get_xticklabels(), rotation=90)
    
    def r2_per_tier(self, plot_args):
        _df = self.test_df.sort_values(['pathway', 'metabolite_id'])
        g: FacetGrid = sns.catplot(
            data=_df, 
            kind='bar',
            x='metabolite_id',
            y='r2',
            hue='strategy',
            palette='deep',
            **plot_args
        )
        g.despine(left=True)
        # g.set(yscale='symlog')

        for axes in g.axes.flat:
            _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=45)

        return g
        # g.set_xticklabels(g.get_xticklabels(), rotation=90)
    
