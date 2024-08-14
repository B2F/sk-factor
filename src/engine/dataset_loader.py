from src.engine.plugins import Plugins

class DatasetLoader():

    @staticmethod
    def load(config, trainfiles):

        loader = config.get('dataset', 'loader')
        return Plugins.create('loader', loader, config, trainfiles).load()
