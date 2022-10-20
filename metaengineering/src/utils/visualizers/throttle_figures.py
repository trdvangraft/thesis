import pandas as pd
import seaborn as sns

class ThorrtleFigures:
    def __init__(self, throttle_df: pd.DataFrame) -> None:
        self.throttle_df = throttle_df
    
    def prediction_per_metabolite(self):
        COL = 'metabolite_id'
    
    def prediction_per_pahtway(self):
        COL = 'pathway'