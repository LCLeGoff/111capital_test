import pandas as pd


class TimeSeriesIndexedByDatesAndSeconds:
    def __init__(self, address):
        self.row_data = pd.read_csv(address)
        self._convert_ts_to_timestamp()

        self.data = self.row_data.copy()
        self.metadata = dict()
        self._reindex_data_by_dates_and_seconds()
        self._initialize_metadata()

    def _convert_ts_to_timestamp(self):
        self.row_data['ts'] = pd.to_datetime(self.row_data['ts'])

    def _reindex_data_by_dates_and_seconds(self):
        self._create_date_column()
        self._create_second_column()
        self._set_dates_and_seconds_as_index()

        self.date_list = list(self.data.index.get_level_values('date').unique())

    def _initialize_metadata(self):
        self.add_to_metadata(
            name='date',
            description='Date of the data (dd/mm/yyyy)',
            index_or_not=True)

        self.add_to_metadata(
            name='seconds',
            description='Time in second passed since the begining of the daily time series (s)',
            index_or_not=True)

        self.add_to_metadata(name='ref_price', description='reference price time series')
        self.add_to_metadata(name='alt_price', description='alternative price time series')
        self.add_to_metadata(name='target', description='target variable, value to predict')

    def _set_dates_and_seconds_as_index(self):
        self.data.set_index(['date', 'seconds'], inplace=True)

    def _create_date_column(self):
        self.data['date'] = self.data['ts'].dt.date

    def _create_second_column(self):
        self.data['seconds'] = self.data['ts'].copy()
        self.data = self.data.groupby('date').apply(self._apply_to_each_group_to_get_seconds)
        self.data['seconds'] = self.data['seconds'].dt.seconds

    def add_to_metadata(self, name, description, index_or_not=False):
        self.metadata[name] = {'name': name, 'description': description, 'index': index_or_not}

    @staticmethod
    def _apply_to_each_group_to_get_seconds(df):
        df['seconds'] -= df['seconds'].iloc[0]
        return df
