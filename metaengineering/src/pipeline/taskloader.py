from dataclasses import dataclass
import pandas as pd
import numpy as np

from anndata import AnnData

from src.settings.tier import Tier
from src.settings.strategy import Strategy

from src.pipeline.config import TaskLoaderConfig



@dataclass
class TaskFrame:
    x: pd.DataFrame
    y: pd.DataFrame
    title: str
    frame_name: str

    def get_data(self):
        return self.x.merge(self.y, left_index=True, right_index=True)

class TaskLoader:
    """
    This is the taskloader docstring
    """

    def __init__(self) -> None:
        self._ann_data_df = None
        self.tl_config = None

    def prepare_taskloader(self, config: TaskLoaderConfig):
        self.tl_config = config

    def prepare_task(self, 
        df: AnnData, 
    ):
        self._ann_data_df = df
        self._build_prepared_frame(self.tl_config)
        return self
    
    def build(self, 
        strategy: Strategy,
        
    ):
        return self._get_model(strategy)
        
    def _get_model(self, 
        strategy: Strategy, 
    ):
        df = self._get_prepared_frame()
        
        if strategy == strategy.ALL:    
            x, y = self._split_frame(df)
            yield TaskFrame(x, y, strategy, 'all')
        elif strategy == strategy.METABOLITE_CENTRIC:
            level = 1
            for v in df.index.unique(level):
                _df = df.xs(v, level=level)
                x, y = self._split_frame(_df)
                yield TaskFrame(x, y, strategy, v)
        elif strategy == strategy.ONE_VS_ALL:
            level = 1
            for v in df.index.unique(level):
                x, y = self._split_frame(df)
                yield TaskFrame(x, y, strategy, v)

    def _throttle_dataframe(self, 
        df: pd.DataFrame, 
        frac: int
    ):
        _df = df.groupby(by='metabolite_id', group_keys=False)
        _df = _df.apply(lambda x: x.sample(frac=frac))
        return _df

    
    def _get_model_by_level(self, level: int, strategy: Strategy):
        df: pd.DataFrame = self._get_prepared_frame()

        for v in df.index.unique(level):
            _df = df.xs(v, level=level)
            x, y = self._split_frame(_df)
            yield TaskFrame(x, y, strategy, v)
    
    def _build_prepared_frame(self, 
        config: TaskLoaderConfig,
    ):
        # we will stack the data along the index
        # basically create a pivot table of the AnnData
        # we need to do that to all frames aligned to the index and columns (basically all elements)
        data = self._ann_data_df.copy()
        x = data.to_df()
        y = data.obs

        if config.tier.value == Tier.TIER0.value:
            df = self._build_prepared_base_frame(x, y)
            df = self._throttle_dataframe(df, config.data_throttle)
        elif config.tier.value == Tier.TIER1.value:
            # TODO we need to generalize the adjancy matrix
            adj_matrix = data.uns['ppi']
            df = self._build_graph_base_frame(x, y, adj_matrix)
        elif config.tier.value == Tier.TIER2.value:
            df = self._build_graph_base_frame(x, y)

        data.uns['prepared_data'] = df

        self._ann_data_df = data
        return self
    
    def _build_graph_base_frame(self, 
        x: pd.DataFrame, 
        y: pd.DataFrame, 
        adj_matrix: pd.DataFrame
    ):
        df = self._get_empty_multi_index_frame(
            indices=[x.index.to_list(), x.columns, y.columns],
            names=['KO_ORF', 'ORF', 'metabolite_id']
        )

        max_enzyme_interaction = int(adj_matrix.sum(axis=1).max())

        # foreach knockout we need to build the adj matrix
        # we need to build in a row based fashion
        r = np.zeros(
            shape=(
                len(x.index) * len(adj_matrix.index), 
                max_enzyme_interaction
            )
        )

        knockout_indices = np.arange(x.shape[0])
        for j in range(adj_matrix.shape[0]):
            indices = np.argwhere(adj_matrix.values[j] > 0).flatten()
            v = x.iloc[knockout_indices, indices].values
            r[(knockout_indices * len(x.index) + j), :v.shape[1]] = v
    

        x = pd.DataFrame(
            data=r,
            index=pd.MultiIndex.from_product(
                [x.index.to_list(), adj_matrix.index],
                names=['KO_ORF', 'ORF']
            ),
            columns=[f"interaction_{i}" for i in range(max_enzyme_interaction)],
        )

        y = self._get_target_frame(y)
        return self._merge_frame(df, x, y)
    
    def _build_prepared_base_frame(self, x: pd.DataFrame, y: pd.DataFrame):
        df = self._get_empty_multi_index_frame(
            indices=[x.index.to_list(), y.columns],
            names=['KO_ORF', 'metabolite_id']
        )
        y = self._get_target_frame(y)
        return self._merge_frame(df, x, y)

    def _get_target_frame(self, y):
        return y.reset_index().melt(
            id_vars=['KO_ORF'], 
            var_name='metabolite_id',
            value_name='metabolite_concentration', 
            ignore_index=False
        ).set_index(['KO_ORF', 'metabolite_id'])
    
    def _get_empty_multi_index_frame(
        self,
        indices,
        names,
    ):
        return pd.DataFrame(
            index=pd.MultiIndex.from_product(
                iterables=indices,
                names=names
            ),
        )
    
    def _merge_frame(self, df, x, y):
        df = df.merge(x, how='left', left_index=True, right_index=True)
        df = df.loc[~(df==0).all(axis=1)]

        df = df.merge(y, how='left', left_index=True, right_index=True)
        df = df.dropna()

        # t = df.loc[("YPR111W", "r5p", "YNL007C")]
        return df

    def _get_prepared_frame(self) -> pd.DataFrame:
        data = self._ann_data_df.copy()
        df: pd.DataFrame = data.uns['prepared_data']
        return df
    
    def _split_frame(self, df: pd.DataFrame):
        x = df.loc[:, df.columns != 'metabolite_concentration']
        y = df['metabolite_concentration']
        return x, y
