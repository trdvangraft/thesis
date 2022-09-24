from dataclasses import dataclass
from functools import reduce
from typing import Callable, List
import pandas as pd

from anndata import AnnData

from src.settings.tier import Tier
from src.settings.task import Tasks
from src.settings.strategy import Strategy

from .dataloader import DataLoader

class TaskTransforms:
    pass

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

    def prepare_task(self, df: AnnData, data_tier: Tier):
        self._ann_data_df = df
        self._build_prepared_frame(data_tier)
        return self
    
    def apply_transform(self, transforms: List[Callable[[AnnData], AnnData]]):
        for transform in transforms:
            self._ann_data_df = transform(self._ann_data_df)
        return self
    
    def build(self, strategy: Strategy):
        if strategy == strategy.ALL:
            return self._get_model_full()
        elif strategy == strategy.METABOLITE_CENTRIC:
            return self._get_model_by_level(1, strategy)
    
    def _get_model_by_level(self, level: int, strategy: Strategy):
        df: pd.DataFrame = self._get_prepared_frame()

        for v in df.index.unique(level):
            x, y = self._split_frame(df.xs(v, level=level))
            yield TaskFrame(x, y, strategy, v)
    
    def _get_model_full(self):
        """
        yield the entire dataframe
        """
        df = self._get_prepared_frame()
        x, y = self._split_frame(df)
        
        yield TaskFrame(x, y, Strategy.ALL, 'all')
    
    def _build_prepared_frame(self, data_tier: Tier):
        # we will stack the data along the index
        # basically create a pivot table of the AnnData
        # we need to do that to all frames aligned to the index and columns (basically all elements)
        data = self._ann_data_df.copy()
        x = data.to_df()
        y = data.obs

        if data_tier.value == Tier.TIER0.value:
            df = self._build_prepared_base_frame(x, y)
        elif data_tier.value == Tier.TIER1.value:
            df = self._build_prepared_simple_frame(x, y)

        data.uns['prepared_data'] = df

        self._ann_data_df = data
        return self
    
    def _build_prepared_simple_frame(self, x: pd.DataFrame, y: pd.DataFrame):
        df = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [x.index.to_list(), x.columns, y.columns],
                names=['KO_ORF', 'ORF', 'metabolite_id']
            ),
        )

        x = x.stack().to_frame('enzyme_concentration')
        y = y.reset_index().melt(
            id_vars=['KO_ORF'], 
            var_name='metabolite_id',
            value_name='metabolite_concentration', 
            ignore_index=False
        ).set_index(['KO_ORF', 'metabolite_id'])

        df = df.merge(x, how='left', left_index=True, right_index=True)
        df = df.merge(y, how='left', left_index=True, right_index=True)
        df = df.dropna()

        return df
    
    def _build_prepared_base_frame(self, x: pd.DataFrame, y: pd.DataFrame):
        df = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [x.index.to_list(), y.columns],
                names=['KO_ORF', 'metabolite_id']
            ),
        )

        y = y.reset_index().melt(
            id_vars=['KO_ORF'], 
            var_name='metabolite_id',
            value_name='metabolite_concentration', 
            ignore_index=False
        ).set_index(['KO_ORF', 'metabolite_id'])

        df = df.merge(x, how='left', left_index=True, right_index=True)
        df = df.merge(y, how='left', left_index=True, right_index=True)
        df = df.dropna()

        return df

    
    def _get_prepared_frame(self) -> pd.DataFrame:
        data = self._ann_data_df.copy()
        df: pd.DataFrame = data.uns['prepared_data']
        return df
    
    def _split_frame(self, df: pd.DataFrame):
        x = df.loc[:, df.columns != 'metabolite_concentration']
        y = df['metabolite_concentration']
        return x, y
