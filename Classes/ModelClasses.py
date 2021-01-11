import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from Classes.TimeSeriesClasses import TimeSeriesIndexedByDatesAndSeconds


class InertiaModels:
    def __init__(self, ts: TimeSeriesIndexedByDatesAndSeconds):
        self.ts = ts
        self.name_to_predict = 'target'

    def daily_prediction(self, date, window):
        lag = int(window / self.ts.dt)
        df = self.ts.data[self.name_to_predict].loc[date, :]

        df = df.rolling(lag).mean()
        df_predicted = df.shift(1)

        return df_predicted

    def prediction(self, window):
        predict_name = 'inertia_w%i_predicted_%s' % (window, self.name_to_predict)
        description = 'Prediction of %s using the inertia model with a window of %i s' % (self.name_to_predict, window)
        self.ts.add_new_columns(name=predict_name, vals=np.nan, description=description)

        for date in self.ts.date_list:
            df_predicted = self.daily_prediction(date, window)
            self.ts.data[predict_name].loc[date, :] = df_predicted.values

    def compute_prediction_of_target_price(self, window):
        predict_name = 'inertia_w%i_predicted_%s' % (window, self.name_to_predict)
        predict_price_name = predict_name+'_price'
        if predict_name not in self.ts.data.columns:
            raise ValueError('Prediction of %s with window %s not done' % (self.name_to_predict, window))
        else:
            df_target_price = self.ts.data[self.name_to_predict + '_price'].shift(1)
            df_predict = self.ts.data[predict_name]
            df_predicted_price = df_target_price + df_predict

            description = 'Prediction of %s price using the inertia model with a window of %i s' \
                          % (self.name_to_predict, window)
            self.ts.add_new_columns(name=predict_price_name, vals=df_predicted_price.copy(), description=description)


class NeuronalNetworkModels:
    def __init__(
            self, model_name, train_ts: TimeSeriesIndexedByDatesAndSeconds, test_ts: TimeSeriesIndexedByDatesAndSeconds,
            feature_names, y_name, layers_sizes=None, n_epochs=10, batch_size=1000, learning_rate=.1):

        self.model_name = model_name
        self.train_ts = train_ts
        self.test_ts = test_ts
        self.feature_names = feature_names
        self.y_name = y_name

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        if layers_sizes is None:
            self.layers_sizes = [len(feature_names), int(len(feature_names))/2.]
        else:
            self.layers_sizes = layers_sizes

        self.train_X = None
        self.train_Y = None
        self.test_X = None
        self.test_Y = None
        self.model = None
        self.iter = None

    def prepare_training_set(self):
        self.train_X = np.zeros((len(self.train_ts.data) - 1, len(self.feature_names)))
        self.train_Y = np.zeros(len(self.train_ts.data) - 1)
        self.iter = 0
        self.train_ts.data.groupby('date').apply(self._prepare_training_set_for_each_day)

    def _prepare_training_set_for_each_day(self, df):

        x_tab = df[self.feature_names].iloc[:-1].to_numpy()
        self.train_X[self.iter:self.iter + self.train_ts.day_length - 1, :] = x_tab

        self.train_Y[self.iter:self.iter + self.train_ts.day_length - 1] = df[self.y_name].iloc[1:].to_numpy()

        self.train_X[np.where(np.isnan(self.train_X))] = 0
        self.train_Y[np.where(np.isnan(self.train_Y))] = 0

        self.iter += self.train_ts.day_length

        return df

    def prepare_test_set(self):
        self.test_X = np.zeros((len(self.test_ts.data) - 1, len(self.feature_names)))
        self.test_Y = np.zeros(len(self.test_ts.data) - 1)
        self.iter = 0
        self.test_ts.data.groupby('date').apply(self._prepare_test_set_for_each_day)
        self.test_X[np.where(np.isnan(self.test_X))] = 0
        self.test_Y[np.where(np.isnan(self.test_Y))] = 0

    def _prepare_test_set_for_each_day(self, df):

        x_tab = df[self.feature_names].iloc[:-1].to_numpy()
        self.test_X[self.iter:self.iter + self.train_ts.day_length - 1, :] = x_tab

        self.test_Y[self.iter:self.iter + self.train_ts.day_length - 1] = df[self.y_name].iloc[1:].to_numpy()

        self.iter += self.train_ts.day_length

        return df

    def build_and_compile_model(self):

        self.model = Sequential()
        self.model.add(
            Dense(self.layers_sizes[0], activation='relu',
                  input_dim=len(self.feature_names)))

        for s in self.layers_sizes[1:]:
            self.model.add(Dense(s, activation='relu'))
        self.model.add(Dense(1, activation='tanh'))

        self.model.compile(loss='mean_squared_error',  metrics=['mean_squared_error'])

    def fit_model(self):
        self.model.fit(x=self.train_X, y=self.train_Y, epochs=self.n_epochs, batch_size=self.batch_size)

    def prediction_on_train(self):

        description = 'Prediction of target with a neuronal network ' \
                      'using the features '+str(self.feature_names)+' with layers of size '+str(self.layers_sizes)\
                      + ' trained for %i epochs and with batch of size %i' % (self.n_epochs, self.batch_size)

        y = self.model.predict(self.train_X).ravel()
        self.train_ts.add_new_columns(name=self.model_name, vals=[np.nan]+list(y), description=description)

    def prediction_on_test(self):
        description = 'Prediction of target with the neuronal network model' \
                      'using the features '+str(self.feature_names)+' with layers of size '+str(self.layers_sizes)\
                      + ' trained for %i epochs and with batch of size %i' % (self.n_epochs, self.batch_size)

        y = self.model.predict(self.test_X).ravel()
        self.test_ts.add_new_columns(name=self.model_name, vals=[np.nan]+list(y), description=description)

    def get_target_price_from_prediction(self):
        predict_price_name = self.model_name+'_price'

        if self.model_name not in self.train_ts.data.columns:
            raise ValueError('Prediction of target not done')
        else:
            for df in [self.train_ts, self.test_ts]:
                description = 'Prediction of target price with the neuronal network model' \
                              'using the features '+str(self.feature_names) + \
                              ' with layers of size '+str(self.layers_sizes)\
                              + ' trained for %i epochs and with batch of size %i' % (self.n_epochs, self.batch_size)
                vals = df.data['target_price'].shift(1) + df.data[self.model_name].values
                df.add_new_columns(name=predict_price_name, vals=vals, description=description)
