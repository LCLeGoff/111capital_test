import pylab as pb

from Classes.ModelClasses import InertiaModels, NeuronalNetworkModels
from Classes.PerformanceClass import PerformanceClass
from Classes.TimeSeriesClasses import TimeSeriesIndexedByDatesAndSeconds

address = 'Data/train.csv'
TrainTS = TimeSeriesIndexedByDatesAndSeconds(address, 'train data set')
address = 'Data/test.csv'
TestTS = TimeSeriesIndexedByDatesAndSeconds(address, 'test data set')

# print(TrainTS.data)
# TrainTS.check_if_day_lengths_and_time_steps_are_equal()
# TrainTS.compute_target_correlation_between_two_days_and_within_a_day()
# TrainTS.compare_daily_target_distribution()
# TrainTS.print_metadata_of('target')

TrainTS.compute_target_price()
TrainTS.compute_smoothing_of_ref_and_alt_price()
TrainTS.compute_first_diff_of_ref_and_alt_price()
TrainTS.compute_normalization_of_target_alt_and_ref_price()
TrainTS.compute_month_weekday_and_month_columns()

TestTS.compute_target_price()
TestTS.compute_smoothing_of_ref_and_alt_price()
TestTS.compute_first_diff_of_ref_and_alt_price()
TestTS.compute_normalization_of_target_alt_and_ref_price()
TestTS.compute_month_weekday_and_month_columns()
# TrainTS.data['normalised_target'].plot()
# TrainTS.data['normalised_alt_price_diff'].plot()
# TrainTS.data['normalised_ref_price_diff'].plot()
# TrainTS.data['smoothed_normalised_alt_price_diff'].plot()
# TrainTS.data['smoothed_normalised_ref_price_diff'].plot()
# pb.show()

# max_lag = 2000
# df_corr = TrainTS.get_mean_correlation_between('normalised_target_price', 'normalised_alt_price')
# df_corr.plot()
# pb.show()
#
#
# InertiaModel = InertiaModels(TrainTS)
# TrainTS.data['target_price'].plot()
# window_list = list(range(10, 610, 10))
#
# for win in window_list:
#     InertiaModel.prediction(win)
#     # InertiaModel.compute_prediction_of_target_price(win)
#     # TrainTS.data['inertia_w%i_predicted_target_price' % win].plot()
#     df_prediction_target = TrainTS.data['inertia_w%s_predicted_target' % win]
#     eucli = PerformanceClass().get_euclidean_distance(TrainTS.data['target'], df_prediction_target)
#     print('%i: %s' % (win, eucli))
#     # loss = PerformanceClass().get_loss('target', win)
#     # print('%i: ' % win, loss)
# pb.legend()
# pb.show()
print()
feature_names = [
    'target', 'target_price',
    'ref_price', 'ref_price_diff', 'smoothed_ref_price',
    'alt_price', 'alt_price_diff', 'smoothed_alt_price',
    'hour', 'month', 'hour'
]
NNModel = NeuronalNetworkModels(
    'nn_model_all_features', TrainTS, TestTS, y_name='target', feature_names=feature_names,
    layers_sizes=[len(feature_names), len(feature_names), int(len(feature_names)/2)], n_epochs=100, batch_size=1000)

NNModel.prepare_training_set()
NNModel.prepare_test_set()
NNModel.build_and_compile_model()
NNModel.fit_model()
NNModel.prediction_on_train()
NNModel.prediction_on_test()

NNModel.get_target_price_from_prediction()

fig, ax = pb.subplots(figsize=(16, 10))
date = TrainTS.date_list[0]

TrainTS.data['target_price'].loc[date, :].loc[date, :].plot(ax=ax, lw=3, c='darkred')

df = TrainTS.data['test_price']
df.loc[date, :].loc[date, :].plot(
    ax=ax, label='Neuronal prediction')

pb.xlim(2300, 3100)
pb.ylim(12, 20)
pb.legend()
pb.show()