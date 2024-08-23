import seaborn as sns
from plugins.plots.base_report import Report
import pandas as pd

class Pairplot(Report):

  def plot(self):

    df = pd.concat(list([self._x, self._y]), axis=1).corr()

    ax = sns.pairplot(
      df,
      hue=self._config.get('preprocess', 'label'),
      kind='scatter',
      diag_kws=None, diag_kind='auto', size=None,
    )

    return 'pairplot'
