from plots.base_report import Report
import matplotlib.pyplot as plt

class Distribution(Report):

    def plot(self):

      n_bins = 5
      if self._config['eda'].get('n_bins'):
          n_bins = self._config['eda']['n_bins']

      if self._config['eda'].get('distribution'):
          feature = eval(self._config['eda']['distribution'])
          plt.hist(self._x[feature], bins=n_bins)
      else:
          plt.hist(self._x, bins=n_bins)

      return 'distribution'
