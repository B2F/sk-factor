from plots.base_report import Report
import matplotlib.pyplot as plt

class DistributionX(Report):

    def plot(self):

        plot_name = 'distribution_x'

        n_bins = 5
        if self._config['eda'].get('n_bins'):
            n_bins = eval(self._config['eda']['n_bins'])

        if self._config['eda'].get('distribution_x'):
            feature = self._config['eda']['distribution_x']
            plt.hist(self._x[feature], bins=n_bins)
            plot_name = 'distribution_' + feature
        else:
            plt.hist(self._x, bins=n_bins)

        return plot_name
