import seaborn as sns
from plots.base_report import Report

class Heatmap(Report):

  def run(self):

    sns.heatmap(
      self._df.corr(),
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

    super().run('heatmap')
