from plugins.plots.base_report import Report
import matplotlib.pyplot as plt

class DistributionX(Report):

    def plot(self):

        plot_name = 'distribution_x'

        n_bins = 5
        if self._config.get('distribution', 'n_bins'):
            n_bins = self._config.get('distribution', 'n_bins')

        if self._config.get('eda', 'distribution_x'):
            plt.hist(self._x[self._config.get('eda', 'distribution_x')], bins=n_bins)
            plot_name = self._config.get('eda', 'distribution_x') + '_distribution'
        else:
            plt.hist(self._x, bins=n_bins)
            plot_name = "_".join(list(self._x.columns)) + ' distribution'

        plt.title(plot_name.replace('_', ' ').title(), pad=20)

        return plot_name
