from typing import List
import pandas as pd
import numpy as np

from scipy.stats import pearsonr

import seaborn as sns


class PredictionFigures():
    def __init__(self, pred_test_df: pd.DataFrame) -> None:
        self.pred_test_df = pred_test_df.sort_values('pathway')

    def prediction_per_pathway_all(self):
        g = sns.scatterplot(
            data=self.pred_test_df,
            x='y_true',
            y='y_pred',
            hue='pathway',
        )

        self._set_axes_limits(g.axes)

    def prediction_per_pathway_individual(self):
        COL = 'pathway'
        self._make_rel_plot(COL)

    def prediction_per_metabolite_individual(self):
        COL = 'metabolite_id'
        self._make_rel_plot(COL)

    def _make_rel_plot(self, col: str):
        rel = self._get_rel(col=col)

        self._set_axes_limits(rel.figure.axes)
        self._draw_diagonals(rel.figure.axes)
        self._draw_mean_prediction(rel.figure.axes, col=col)
        self._draw_r2(rel.figure.axes, col=col)

    def _set_axes_limits(self, axes, limits: List = [-1.5, 1.5]):
        if type(axes) == list:
            for ax in axes:
                ax.set_xlim(limits)
                ax.set_ylim(limits)
        else:
            axes.set_xlim(limits)
            axes.set_ylim(limits)

    def _draw_diagonals(self, axes, limits: List = [-1.5, 1.5]):
        for ax in axes:
            ax.plot(limits, limits, ls='--', c='black', alpha=0.8, lw=0.7)

    def _draw_mean_prediction(self, axes, col: str, limits: List = [-1.5, 1.5]):
        for ax, value in zip(axes, self.pred_test_df[col].unique()):
            mean = np.mean(self.pred_test_df[self.pred_test_df[col] == value]['y_true'])
            ax.axhline(y=mean, xmin=limits[0], xmax=limits[1], alpha=0.8, lw=0.7, c='red', ls='--')

    def _draw_r2(self, axes, col: str):
        for ax, pathway in zip(axes, self.pred_test_df[col].unique()):
            _cdf = self.pred_test_df[self.pred_test_df[col] == pathway]
            r, p = pearsonr(_cdf['y_true'], _cdf['y_pred'])
            ax.text(
                .05, .8,
                f'r={r:.2f}, p={p:.2g}',
                transform=ax.transAxes
            )

    def _get_rel(self, col: str):
        rel = sns.relplot(
            data=self.pred_test_df,
            x="y_true",
            y="y_pred",
            kind="scatter",
            col=col,
            hue='pathway',
            col_wrap=min(4, len(self.pred_test_df[col].unique())),
            facet_kws={'sharey': False, 'sharex': False}
        )
        return rel
