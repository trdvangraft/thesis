from typing import List, Callable

from anndata import AnnData
from src.pipeline.config import ParsedDataLoaderConfig, DataLoaderConfig

from src.pipeline.frame.cache import FrameCache
from src.pipeline.frame.transformer import FrameTransformers
from src.pipeline.frame.loader import FrameLoaders
from src.pipeline.frame.filter import FrameFilters

class DataFactory:
    def __init__(self, root_dir: str) -> None:
        self.frame_cache: FrameCache = FrameCache()
        self._transformer: FrameTransformers = FrameTransformers(self.frame_cache)
        self._loaders: FrameLoaders = FrameLoaders(self.frame_cache, root_dir)
        self._filters: FrameFilters = FrameFilters(self.frame_cache)

    @property
    def loaders(self):
        return self._loaders
    
    @property
    def transformer(self):
        return self._transformer
    
    @property
    def filters(self):
        return self._filters
    
    def parse_config(self, config: DataLoaderConfig) -> ParsedDataLoaderConfig:
        return ParsedDataLoaderConfig(
            additional_filters=self.filters.parse_config(config.additional_filters),
            additional_frames=self.loaders.parse_config(config.additional_frames),
            additional_transforms=self.transformer.parse_config(config.additional_transforms)
        )

    def load(self, frames: List[Callable]):
        for frame_loader in frames:
            frame_loader()
        return self

    def transform(self, transforms: List[Callable]):
        for transform in transforms:
            transform()
        return self
    
    def filter(self, filters: List[Callable] = []):
        for filter in filters:
            filter()
        return self

    def build(self):
        self.current_frame = AnnData(
            X=self.frame_cache.get_frame('proteins'), 
            obs=self.frame_cache.get_frame('metabolites'),
            dtype=self.frame_cache.get_frame('proteins').dtypes,
        )

        if self.frame_cache.contains('protein_expression'):
            for key, item in self.frame_cache.get_frame('protein_expression').get_all_frames():
                self.current_frame.varm[key] = item

        if self.frame_cache.contains('ppi'):
            ppi = self.frame_cache.get_frame('ppi')
            self.current_frame.uns['ppi'] = ppi

        df = self.current_frame.copy()
        self.current_frame = None
        return df