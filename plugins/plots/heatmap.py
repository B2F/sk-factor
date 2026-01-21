import seaborn as sns
from plugins.plots.base_report import Report
import pandas as pd
from src.utils.data_validator import validate_numeric_columns

class Heatmap(Report):

  def plot(self):
    # Check for non-numeric columns that could cause conversion errors
    df_combined = pd.concat(list([self._x, self._y]), axis=1)
    validate_numeric_columns(df_combined, "Heatmap correlation matrix")

    df = df_combined.corr()

    ax = sns.heatmap(
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

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

    return 'heatmap'
