import seaborn as sns
from plugins.plots.base_report import Report
import matplotlib.pyplot as plt

class Pairplot(Report):

  def run(self):

    sns.pairplot(
      self._x,
      hue='Plot',
      kind='hist',
      # palette = ["#0000ff", "#55aa00", "#005500", "#ff0000"],
      # markers=['o', 'D', 'D', 'o'],
      diag_kws=None, diag_kind='hist', size=None,
      # height=3,
    )

    return 'pairplot'
