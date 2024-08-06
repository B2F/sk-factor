import seaborn as sns
from plots.base_report import Report
import pandas as pd

class Heatmap(Report):

  def plot(self):

    df = pd.concat(list([self._x, self._y]), axis=1).corr()

    sns.heatmap(
      df,
      cmap = 'coolwarm',
      annot=True,
      vmin = -1,
      vmax = 1,
      center = 0,
      fmt=".2f",
      square=True,
      linewidths=.5,
      xticklabels=True,
      yticklabels=True,
    )

    return 'heatmap'
