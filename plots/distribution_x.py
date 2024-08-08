from plots.base_report import Report
import matplotlib.pyplot as plt

class DistributionX(Report):

    def plot(self):

      n_bins = 5
      if self._config['eda'].get('n_bins'):
          n_bins = eval(self._config['eda']['n_bins'])

      if self._config['eda'].get('distribution_x'):
          feature = self._config['eda']['distribution_x']
          plt.hist(self._x[feature], bins=n_bins)
      else:
          plt.hist(self._x, bins=n_bins)

      return 'distribution_x'
