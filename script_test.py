from Classes.TimeSeriesClasses import TimeSeriesIndexedByDatesAndSeconds

address = 'Data/train.csv'
TrainTS = TimeSeriesIndexedByDatesAndSeconds(address)
print(TrainTS.data)
