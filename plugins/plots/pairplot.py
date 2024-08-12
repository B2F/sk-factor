import seaborn as sns
from plugins.plots.base_report import Report
import pandas as pd

class Pairplot(Report):

  def plot(self):

    df = pd.concat(list([self._x, self._y]), axis=1).corr()

    g = sns.pairplot(
      df,
      hue=self._config['preprocess']['label'],
      kind='scatter',
      # palette = ["#0000ff", "#55aa00", "#005500", "#ff0000"],
      # markers=['o', 'D', 'D', 'o'],
      diag_kws=None, diag_kind='auto', size=None,
      # height=3,
    )

    return 'pairplot'
