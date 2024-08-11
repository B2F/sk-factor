from plugins.plots.base_report import Report

class MeanLabel(Report):

  def run(self):

    print('\nCSV shape:')
    print(self._x.shape)

    # https://www.geeksforgeeks.org/regression-using-lightgbm/
    for i, col in enumerate(self._x.columns):
      if col == 'Plot':
        continue
      print('\n' + col)
      grouped_data = self._x[['Plot', col]].groupby('Plot').mean()
      print('max: ' + str(max(self._x[col])) )
      print ('min: ' + str(min(self._x[col])) )
      print('\n')
      print (grouped_data)
