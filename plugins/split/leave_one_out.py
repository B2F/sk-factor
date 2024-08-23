from sklearn.model_selection import LeaveOneOut as BaseLeaveOneOut
from plugins.split.base_cv import BaseCv

class LeaveOneOut(BaseCv):

    def split(self):
        l = BaseLeaveOneOut()
        return l.split(self._x, self._y)
