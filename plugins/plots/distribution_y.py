from plugins.plots.base_report import Report
import matplotlib.pyplot as plt

class DistributionY(Report):

    def plot(self):

      n_bins = 5
      if self._config['eda'].get('n_bins'):
          n_bins = eval(self._config['eda']['n_bins'])

      plt.hist(self._y, bins=n_bins)

      return 'distribution_y'
