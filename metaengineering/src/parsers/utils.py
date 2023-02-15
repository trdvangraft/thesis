from src.settings.tier import Tier
from src.settings.strategy import Strategy

def map_strategy(x):
    mapping = {
        Strategy.ALL: 'All metabolite', 
        Strategy.ONE_VS_ALL: 'Leave one metabolite out', 
        Strategy.METABOLITE_CENTRIC: 'Single metabolite',
        'all': 'All metabolite',
        'one_vs_all': 'Leave one metabolite out',
        'metabolite': 'Single metabolite',
    }
    return mapping[x]

def map_tier(x):
    mapping = {
        Tier.TIER0: 'Baseline', 
        Tier.TIER1: 'PPI', 
        Tier.TIER2: 'Stiochiometric protein',
        Tier.TIER3: 'Stiochiometric metabolites',
    }
    return mapping[x]

def map_architecture(x):
    mapping = {
        "SVR()": 'SVR', 
        "RandomForestRegressor()": 'Random Forest', 
        "ElasticNet()": 'Elastic net',
        "MLPRegressor()": 'MLP'
    }
    return mapping[x]