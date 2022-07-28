from src.models import BaseModel

import statsmodel.api as sm

class GLM(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.title = "Generalized linear model"
        self.model = None
    
    def _create_model(self, x, y, **kwargs):
        self.model = sm.GLM()
    
    def fit():
        pass
