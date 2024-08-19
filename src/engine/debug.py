import importlib

class Debugger():

    @staticmethod
    def attach(config):

        if config.eq('debug', 'enabled', False):
            return

        debugpy = importlib.import_module("debugpy")
        debugpy.listen((config.get('debug', 'host'), config.get('debug', 'port')))
        if config.eq(*('debug', 'wait_for_client'), True):
            debugpy.wait_for_client()
