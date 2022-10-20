from typing import Union, Dict, Any

import pandas as pd

class FrameCache:
    def __init__(self) -> None:
        self.frame_cache: Dict[str, Union[pd.DataFrame, FrameCache]] = {}

    def insert_frame(self, name: str, frame: pd.DataFrame):
        self.frame_cache[name] = frame.copy()
    
    def get_frame(self, name, default: Any=None):
        return self.frame_cache.get(name, default)
    
    def update_frame(self, name: str, frame: Union[pd.DataFrame, Any]):
        if type(frame) == pd.DataFrame:
            self.frame_cache[name] = frame.copy()
        elif type(frame) == FrameCache:
            self.frame_cache[name] = frame
    
    def get_all_frames(self):
        return self.frame_cache.items()
    
    def contains(self, name: str):
        return name in self.frame_cache