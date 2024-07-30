import random
from pipeline.base_estimator import BaseEstimator

class BaseLgbm(BaseEstimator):

    _params: dict

    def __init__(self, config):

        if config['training']['seed'] is int:
            seed = config['training']['seed']
        elif config['training']['seed'] == 'random':
            seed = random.seed()

        self._params = {
          "boosting_type": "gbdt",
          "num_leaves": 31,
          "max_depth": -1,
          "learning_rate": 0.1,
          "n_estimators": 100,
          "seed": seed
        }
