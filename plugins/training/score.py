from plugins.training.base_runner import BaseRunner
from sklearn.model_selection import cross_val_score

class Score(BaseRunner):

    def run(self, pipeline, cv):

        scoring = 'accuracy'
        if self._config.get('training', 'scoring'):
            scoring = self._config.get('training', 'scoring')

        scores = cross_val_score(
            pipeline,
            self._x,
            self._y.values.flatten(),
            cv = cv,
            scoring = scoring,
            n_jobs=-1
        )

        print(scores)
