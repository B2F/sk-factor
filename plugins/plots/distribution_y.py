from plugins.plots.base_report import Report
import matplotlib.pyplot as plt

class DistributionY(Report):

    def plot(self):

        n_bins = 5
        if self._config.get('eda', 'n_bins'):
            n_bins = self._config.get('eda', 'n_bins')

        plt.hist(self._y, bins=n_bins)

        plt.title('Label distribution', pad=20)

        return 'distribution_y'
