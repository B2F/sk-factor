from eda_plots.base_report import Report

class MeanLabel(Report):

  def run(self):

    print('\nCSV shape:')
    print(self._df.shape)

    # https://www.geeksforgeeks.org/regression-using-lightgbm/
    for i, col in enumerate(self._df.columns):
      if col == 'performance':
        continue
      print('\n' + col)
      grouped_data = self._df[['performance', col]].groupby('performance').mean()
      print('max: ' + str(max(self._df[col])) )
      print ('min: ' + str(min(self._df[col])) )
      print('\n')
      print (grouped_data)
