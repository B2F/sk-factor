from sklearn.ensemble import HistGradientBoostingRegressor
from plugins.estimators.base_estimator import BaseEstimator

class Hgbr(BaseEstimator):

    def getEstimator(
        self,
        loss='squared_error',
        quantile=None,
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        max_depth=None,
        min_samples_leaf=20,
        l2_regularization=0.0,
        max_features=1.0,
        max_bins=255,
        categorical_features='warn',
        monotonic_cst=None,
        interaction_cst=None,
        warm_start=False,
        early_stopping='auto',
        scoring='loss',
        validation_fraction=0.1,
        n_iter_no_change=10,
        tol=1e-07,
        verbose=0,
    ):

        regressor = HistGradientBoostingRegressor(
            loss=loss,
            quantile=quantile,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_features=max_features,
            max_bins=max_bins,
            categorical_features=categorical_features,
            monotonic_cst=monotonic_cst,
            interaction_cst=interaction_cst,
            warm_start=warm_start,
            early_stopping=early_stopping,
            scoring=scoring,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            verbose=verbose,
            random_state=self._config.get('training', 'seed'),
        )

        return ('hist_gradient_boosting_regressor', regressor)
