import pandas as pd

import seaborn as sns

class CVFigures():
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
    
    def cv_results(self):
        for reggressor in self.df['regressor'].unique():
            _df = self.df[self.df['regressor'] == reggressor].sort_values(['pathway', 'metabolite_id'])
            print(_df.shape)
            g = sns.catplot(
                x='metabolite_id',
                y='mean',
                hue='pathway',
                col='params_fmt',
                col_wrap=6,
                sharey=True,
                sharex=False,
                kind='bar',
                data=_df,
                dodge=False,
                palette='deep',
                # legend=False,
            )

            # g.set(yscale='symlog')
            g.set_xticklabels(rotation=90)
            # plt.subplots_adjust(hspace=0.4)