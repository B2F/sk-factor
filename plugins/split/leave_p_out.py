from sklearn.model_selection import LeavePOut as BaseLeavePOut
from plugins.split.base_cv import BaseCv

class LeavePOut(BaseCv):

    def split(self):

        p_out=2
        if self._config.get('split', 'p_out'):
            p_out = self._config.get('split', 'p_out')

        l = BaseLeavePOut(p=p_out)
        return l.split(self._x, self._y, self._groups)
