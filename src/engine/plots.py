from src.engine.plugins import Plugins

class Plots():

    @staticmethod
    def run(plots, config, x, y, y_labels, identifier):

        if plots is None:
            return

        for plot in plots:
            args = (config, x, y, y_labels, identifier)
            plotObj = Plugins.create('plots', plot, *args)
            plotObj.run()
