import numpy as np


class PerformanceClass:
    def __init__(self):
        pass

    @staticmethod
    def get_euclidean_distance(df_real, df_prediction):
        n = len(df_prediction.dropna())
        res = np.sqrt(np.nansum((df_real - df_prediction) ** 2)) / n
        return res

    @staticmethod
    def get_loss(df_real, df_real_price, df_prediction, df_prediction_price, day_length):

        n_money_real = np.nanmean(df_real_price * df_prediction)*day_length
        n_money_prediction = np.nanmean(df_prediction_price * df_prediction)*day_length
        n_money_optimal = np.nanmean(df_real_price * df_real)*day_length

        return n_money_prediction-n_money_real, n_money_optimal
