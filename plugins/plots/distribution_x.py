from plugins.plots.base_report import Report
import matplotlib.pyplot as plt

class DistributionX(Report):

    def plot(self):

        plot_name = 'distribution_x'

        n_bins = 5
        if self._config.get('eda', 'n_bins'):
            n_bins = self._config.get('eda', 'n_bins')

        if self._config.get('eda', 'distribution_x'):
            feature = self._config.get('eda', 'distribution_x')
            plt.hist(self._x[feature], bins=n_bins)
            plot_name = 'distribution_' + feature
        else:
            plt.hist(self._x, bins=n_bins)

        return plot_name
