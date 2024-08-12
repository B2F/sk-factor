import pandas as pd
from plugins.pipeline.base_pipeline import BasePipeline
from src.engine.plugins import Plugins
from src.engine.config import Config

class Pipeline():

    @staticmethod
    def create(estimators: list, x: pd.DataFrame, y: pd.DataFrame, config: Config):

        # Assemble all estimators (sampling, classifiers ...) in a single pipeline
        factoredPipeline = BasePipeline(config)
        for estimator in config.get('training', 'estimators'):
            # Put GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV etc in the pipeline directiory.
            # Same for automated feature selection ? (RFECV, SelectFromModel)
            estimator = Plugins.create('pipeline', estimator, config, x, y)
            factoredPipeline.addStep(estimator.getEstimator())

        return factoredPipeline.getPipeline()
