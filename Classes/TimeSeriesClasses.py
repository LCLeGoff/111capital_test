import pandas as pd
import numpy as np
import pylab as pb
import scipy.stats as scs


class TimeSeriesIndexedByDatesAndSeconds:
    def __init__(self, address, description):
        self.description = description
        self.row_data = pd.read_csv(address)
        self._convert_ts_to_timestamp()

        self.data = self.row_data.copy()
        self.metadata = {}
        self._initialize_metadata()
        self._reindex_data_by_dates_and_t()
        self._create_i_columns()

        self.date_list = None
        self.day_length = None
        self.dt = None
        self._get_date_list_and_day_length_and_timestep()

        self.iter = None

    def _add_to_metadata(self, name, description):
        self.metadata[name] = {'description': description}

    def add_new_columns(self, name, vals, description):
        self._add_to_metadata(name=name, description=description)
        self.data[name] = vals

    def print_description(self):
        print(self.description)

    def print_metadata_of(self, name):
        print('Metadata on %s' % name)
        print('  Description: %s' % self.metadata[name]['description'])

    def compute_first_diff_of_ref_and_alt_price(self):
        name = 'ref_price_diff'
        description = 'First difference of the reference price, ' \
                      'i.e. D_i = P_i-P_{i-1}, if P_i is the price and D_i the first difference at time step i'
        vals = self.data['ref_price'] - self.data['ref_price'].shift(1).values
        self.add_new_columns(name=name, vals=vals, description=description)

        name = 'alt_price_diff'
        description = 'First difference of the alternative price, ' \
                      'i.e. D_i = P_i-P_{i-1}, if P_i is the price and D_i the first difference at time step i'
        vals = self.data['alt_price'] - self.data['alt_price'].shift(1).values
        self.add_new_columns(name=name, vals=vals, description=description)

    def compute_smoothing_of_ref_and_alt_price(self):
        name = 'smoothed_ref_price'
        description = 'Reference price smoothed by a moving average of window 590 seconds'
        self.add_new_columns(name=name, vals=np.nan, description=description)

        name = 'smoothed_alt_price'
        description = 'Alternative price smoothed by a moving average of window 590 seconds'
        self.add_new_columns(name=name, vals=np.nan, description=description)

        window = int(590/self.dt)

        def _smooth_each_group(df):
            df['smoothed_ref_price'] = df['ref_price'].rolling(window).mean()
            df['smoothed_alt_price'] = df['alt_price'].rolling(window).mean()
            return df

        self.data = self.data.groupby('date').apply(_smooth_each_group)

    def compute_target_price(self):
        name = 'target_price'
        description = 'Cumulative sum of the target variable, ' \
                      'i.e. S_i = S_i-1+T_i and S_0 = T0,' \
                      ' if S_i is the cumulative sum and T_i the target at time step i'
        vals = self.data['target'].cumsum()
        self.add_new_columns(name=name, vals=vals, description=description)

    def _convert_ts_to_timestamp(self):
        self.row_data['ts'] = pd.to_datetime(self.row_data['ts'])

    def _reindex_data_by_dates_and_t(self):
        self._create_date_column()
        self._create_t_column()
        self._set_dates_and_seconds_as_index()

        self.date_list = list(self.data.index.get_level_values('date').unique())

    def _create_i_columns(self):
        name = 'i'
        description = 'Time step since the first entry of the data set.' \
                      ' The real time between to time step can be 10 seconds within the same day or several hours'
        vals = range(len(self.data))
        self.add_new_columns(name=name, vals=vals, description=description)

    def _initialize_metadata(self):
        self._add_to_metadata(name='ref_price', description='reference price time series')
        self._add_to_metadata(name='alt_price', description='alternative price time series')
        self._add_to_metadata(name='target', description='target variable, value to predict')

    def _get_date_list_and_day_length_and_timestep(self):
        self.date_list = list(self.data.index.get_level_values('date').unique())
        self.date_list.sort()

        self.day_length = len(self.data.loc[pd.IndexSlice[self.date_list[0], :], :])

        self.dt = 10

    def _set_dates_and_seconds_as_index(self):
        self.data.set_index(['date', 't'], inplace=True)

    def _create_date_column(self):
        self.add_new_columns(
            name='date',
            vals=self.data['ts'].dt.date,
            description='Date of the data (dd/mm/yyyy)')

    def _create_t_column(self):
        name = 't'
        description = 'Time in second passed since the begining of the daily time series (s)'
        vals = self.data['ts'].copy()
        self.add_new_columns(name=name, vals=vals, description=description)

        self.data = self.data.groupby('date').apply(self._apply_to_each_day_to_get_t)
        self.data[name] = self.data[name].dt.seconds

    @staticmethod
    def _apply_to_each_day_to_get_t(df):
        df['t'] -= df['t'].iloc[0]
        return df

    def check_if_day_lengths_and_time_steps_are_equal(self):
        list_dt = []
        list_day_lengths = []
        for date in self.date_list:
            df = self.data.loc[date, :]

            tab_t = np.array(df.index.get_level_values('t'))
            list_dt += list(tab_t[1:]-tab_t[:-1])

            list_day_lengths.append(len(df))

        dt = list_dt[0]
        if any(np.array(list_dt) != dt):
            print('Not all time steps are equal')
        else:
            print('All time steps are equal')

        length = list_day_lengths[0]
        if any(np.array(list_day_lengths) != length):
            print('Days do not have the same length')
        else:
            print('Days have the same length')

    def compute_target_correlation_between_two_days_and_within_a_day(self):
        target_name = 'target'

        list_day_end = []
        list_day_start = []
        list_rest_of_the_day0 = []
        list_rest_of_the_day1 = []

        date = self.date_list[0]
        list_day_end.append(self.data[target_name].loc[date, :].iloc[-2])

        for date in self.date_list[1:-1]:
            df = self.data[target_name].loc[date, :]
            list_day_start.append(df.iloc[0])
            list_day_end.append(df.iloc[-2])

            df = df.iloc[1:-1]
            list_rest_of_the_day0 += list(df.shift(1).dropna().values)
            list_rest_of_the_day1 += list(df.iloc[1:].dropna().values)

        date = self.date_list[-1]
        list_day_start.append(self.data[target_name].loc[date, :].iloc[0])

        corr = scs.pearsonr(list_day_end, list_day_start)
        print('correlation between days %f:' % corr[0])

        corr = scs.pearsonr(list_rest_of_the_day0, list_rest_of_the_day1)
        print('correlation within a day %f:' % corr[0])

    def compare_daily_target_distribution(self):
        target_name = 'target'
        for i in range(len(self.date_list)-1):
            date0 = self.date_list[i]
            date1 = self.date_list[i+1]

            df0 = self.data[target_name].loc[date0, :]
            df1 = self.data[target_name].loc[date1, :]

            print(scs.ks_2samp(df0.values, df1.values))
        lim = int(len(self.data) / 2)
        print(scs.ks_2samp(self.data[target_name].iloc[:lim].values, self.data[target_name].iloc[lim:].values))

    @staticmethod
    def get_daily_correlation_between(name1, name2, df, max_lag):
        res_corr = np.full(2 * max_lag + 1, np.nan)
        df1 = df[name1]
        df2 = df[name2]
        for ii, lag in enumerate(range(-max_lag, max_lag + 1)):
            temp = pd.concat([df1, df2.shift(lag)], axis=1)
            res_corr[ii] = temp.corr().iloc[0, 1]
        return res_corr

    def get_mean_correlation_between(self, name1, name2, max_lag=None):
        if max_lag is None:
            max_lag = self.day_length
        else:
            max_lag = int(max_lag/self.dt)

        res_corr = np.zeros(2 * max_lag - 1)

        def _get_corr_for_each_group(df):

            x1 = np.array(df[name1]).ravel()
            std1 = np.nanstd(x1)
            x1 = x1 - np.nanmean(x1)

            x2 = np.array(df[name2]).ravel()
            std2 = np.nanstd(x2)
            x2 = x2 - np.nanmean(x2)

            mask = np.where(np.isnan(x1) | np.isnan(x1))[0]
            x1[mask] = 0
            x2[mask] = 0

            norm = np.array(list(range(1, len(x1)+1, 1))+list(range(len(x1)-1, 0, -1)))
            r = np.correlate(x1, x2, mode='full')
            res_corr[:] += r/std1/std2/norm
            return df

        self.data.groupby('date').apply(_get_corr_for_each_group)

        res_corr /= len(self.date_list)
        t = range(-(max_lag-1)*self.dt, max_lag*self.dt-1, self.dt)

        if name1 == name2:
            name = 'autocorr_'+name1
        else:
            name = 'corr_btw_%s_and_%s' % (name1, name2)

        df_corr = pd.DataFrame(index=t, data=res_corr, columns=[name])

        return df_corr

    def compute_month_weekday_and_month_columns(self):
        name = 'hour'
        description = 'Hour'
        vals = self.row_data.ts.dt.hour.values
        self.add_new_columns(name=name, vals=vals, description=description)

        name = 'weekday'
        description = 'Week day, i.e. 0 is Monday, 1 is Tuesday, ... and 6 is Sunday'
        vals = self.row_data.ts.dt.dayofweek.values
        self.add_new_columns(name=name, vals=vals, description=description)

        name = 'month'
        description = 'Month number'
        vals = self.row_data.ts.dt.month.values
        self.add_new_columns(name=name, vals=vals, description=description)

    def plot_histogram_of_target_by_weekday_month_and_hour(self):

        dx = 0.0125*3
        bins = np.arange(0, 1+dx, dx)
        x = (bins[1:]+bins[:-1])/2.
        for name in ['hour', 'dayofweek', 'month']:
            vals = self.data[name].unique()
            vals.sort()
            for val in vals:
                y, _ = np.histogram(self.data['target'][self.data[name] == val].abs().values, bins, density=True)
                pb.plot(x, y, label=val)
            pb.title(name)
            pb.legend()
            pb.show()

    def compute_normalization_of_target_alt_and_ref_price(self):

        self._compute_normalisation_of_target_and_first_diff_of_ref_and_alt_price()
        self._compute_normalized_cumsum_of_target_and_ref_and_alt_price()
        self._compute_smoothing_of_normalised_target_cumsum_and_ref_and_target_price()
        self._compute_first_diff_of_normalised_ref_and_alt_price()

    def _compute_smoothing_of_normalised_target_cumsum_and_ref_and_target_price(self):
        name = 'smoothed_normalised_ref_price'
        description = 'Normalised reference price smoothed by a moving average of window 590 seconds'
        self.add_new_columns(name=name, vals=np.nan, description=description)
        name = 'smoothed_normalised_alt_price'
        description = 'Normalised alternative price smoothed by a moving average of window 590 seconds'
        self.add_new_columns(name=name, vals=np.nan, description=description)
        window = int(590 / self.dt)

        def _smooth_each_group(df):
            df['smoothed_normalised_ref_price'] = df['normalised_ref_price'].rolling(window).mean()
            df['smoothed_normalised_alt_price'] = df['normalised_alt_price'].rolling(window).mean()
            return df

        self.data = self.data.groupby('date').apply(_smooth_each_group)

    def _compute_normalized_cumsum_of_target_and_ref_and_alt_price(self):
        name = 'normalised_target_price'
        description = 'Cumulative sum of the normalised target variable, ' \
                      'i.e. S_i = S_i-1+T_i and S_0 = T0,' \
                      ' if S_i is the cumulative sum and T_i the target at time step i'
        vals = self.data['normalised_target'].cumsum()
        self.add_new_columns(name=name, vals=vals, description=description)
        name = 'normalised_ref_price'
        description = 'Cumulative sum of the normalised reference price, ' \
                      'i.e. S_i = S_i-1+T_i and S_0 = T0,' \
                      ' if S_i is the cumulative sum and T_i the target at time step i'
        vals = self.data['normalised_ref_price_diff'].cumsum()
        self.add_new_columns(name=name, vals=vals, description=description)
        name = 'normalised_alt_price'
        description = 'Cumulative sum of the normalised alternative price, ' \
                      'i.e. S_i = S_i-1+T_i and S_0 = T0,' \
                      ' if S_i is the cumulative sum and T_i the target at time step i'
        vals = self.data['normalised_alt_price_diff'].cumsum()
        self.add_new_columns(name=name, vals=vals, description=description)

    def _compute_normalisation_of_target_and_first_diff_of_ref_and_alt_price(self):
        for name in ['ref_price_diff', 'alt_price_diff']:
            norm_name = 'normalised_' + name
            description = '%s normalised,' \
                          ' i.e. %s is divided by the standard deviation of its smoothing ' \
                          'during the previous hour' % (name, name)
            vals = self.data[name].copy()
            self.add_new_columns(name=norm_name, vals=vals, description=description)
        name = 'target'
        norm_name = 'normalised_' + name
        description = '%s normalised,' \
                      ' i.e. %s is divided by its standard deviation of its smoothing ' % (name, name)
        vals = self.data[name].copy()
        self.add_new_columns(name=norm_name, vals=vals, description=description)

        def _normalise_for_each_day(df):
            std_ref = df['smoothed_ref_price'].dropna().std()
            std_alt = df['smoothed_alt_price'].dropna().std()
            std_target = df['target_price'].dropna().std()

            df['normalised_target'] /= std_target
            df['normalised_ref_price_diff'] /= std_ref
            df['normalised_alt_price_diff'] /= std_alt

            df['normalised_target'].iloc[:60] = np.nan
            df['normalised_ref_price_diff'].iloc[:60] = np.nan
            df['normalised_alt_price_diff'].iloc[:60] = np.nan
            return df

        self.data = self.data.groupby('date').apply(_normalise_for_each_day)

    def _compute_first_diff_of_normalised_ref_and_alt_price(self):
        name = 'smoothed_normalised_ref_price_diff'
        description = 'First difference of the smoothed and normalised reference price, ' \
                      'i.e. D_i = P_i-P_{i-1}, if P_i is the price and D_i the first difference at time step i'
        vals = self.data['smoothed_normalised_ref_price'] - self.data['smoothed_normalised_ref_price'].shift(1).values
        self.add_new_columns(name=name, vals=vals, description=description)

        name = 'smoothed_normalised_alt_price_diff'
        description = 'First difference of the smoothed and normalised price, ' \
                      'i.e. D_i = P_i-P_{i-1}, if P_i is the price and D_i the first difference at time step i'
        vals = self.data['smoothed_normalised_alt_price'] - self.data['smoothed_normalised_alt_price'].shift(1).values
        self.add_new_columns(name=name, vals=vals, description=description)
