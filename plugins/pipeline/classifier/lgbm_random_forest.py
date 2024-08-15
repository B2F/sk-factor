import lightgbm as lgb
from plugins.pipeline.base_estimator import BaseEstimator

class LgbmRandomForest(BaseEstimator):

    _type = 'classifier'

    def getEstimator(
        self,
        boosting_type = "gbdt",
        num_leaves = 31,
        max_depth = -1,
        learning_rate = 0.1,
        n_estimators = 100,
        subsample_for_bin = 200000,
        objective = None,
        class_weight= None,
        min_split_gain = 0.0,
        min_child_weight = 1e-3,
        min_child_samples = 20,
        subsample = 1.0,
        subsample_freq = 0,
        colsample_bytree = 1.0,
        reg_alpha = 0.0,
        reg_lambda = 0.0,
        random_state = None,
        n_jobs = None,
        importance_type = "split",
    ):

        classifier = lgb.LGBMClassifier(
            verbosity=-1,
            boosting_type = boosting_type,
            num_leaves = num_leaves,
            max_depth = max_depth,
            learning_rate = learning_rate,
            n_estimators = n_estimators,
            subsample_for_bin = subsample_for_bin,
            objective = objective,
            class_weight = class_weight,
            min_split_gain = min_split_gain,
            min_child_weight = min_child_weight,
            min_child_samples = min_child_samples,
            subsample = subsample,
            subsample_freq = subsample_freq,
            colsample_bytree = colsample_bytree,
            reg_alpha = reg_alpha,
            reg_lambda = reg_lambda,
            random_state = random_state,
            n_jobs = n_jobs,
            importance_type = importance_type,
        )

        return ('lgbm_random_forest', classifier)
