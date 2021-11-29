#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# shiftwidth=2

# -------------> imports
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
from src.calculate_AQI import calculate_AQI
from src.WindowGenerator import WindowGenerator
from src.Baseline import Baseline
from src.compile_and_fit import compile_and_fit
from src.FeedBack import FeedBack

# -------------> configuration
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# -------------> data wrangling
data1_A_hourly_real = pd.read_excel("/home/competition/math_modeling/questions/B/1.xlsx", sheet_name="监测点A逐小时污染物浓度与气象实测数据", na_values='—')
df = data1_A_hourly_real.copy()
#  data2_C_hourly_real = pd.read_excel("/home/competition/math_modeling/questions/B/2.xlsx", sheet_name="监测点C逐小时污染物浓度与气象实测数据", na_values='—')
#  df = data2_C_hourly_real.copy()
date_time = pd.to_datetime(df.pop('监测时间'), format='%Y-%m-%d %H:%M:%S')
df.pop('地点')

# -------------> null processing
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp_mean = IterativeImputer(random_state=0)
imp_mean.fit(df.values)
imp = imp_mean.transform(df.values)
df_imp = pd.DataFrame(imp, columns=df.columns)
df = df_imp

# -------------> wind direction transform: waiting
wv = df.pop('风速(m/s)')
wd_rad = df.pop('风向(°)')*np.pi / 180
df['Wx'] = wv*np.cos(wd_rad)
df['Wy'] = wv*np.sin(wd_rad)

# -------------> time transform: waiting for half day
timestamp_s = date_time.map(pd.Timestamp.timestamp)
day = 24*60*60
year = (365.2425)*day
df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

# -------------> dataset split
column_indices = {name: i for i, name in enumerate(df.columns)}
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]
num_features = df.shape[1]

pred_df = df[int(n*0.99):]
# ----------------------------------------------------------------


# -------------> normalization: waiting for moving average
train_mean = train_df.mean()
train_std = train_df.std()
train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

pred_mean = pred_df.mean()
pred_std = pred_df.std()
pred_df = (pred_df - pred_mean) / pred_std
# ----------------------------------------------------------------

# -------------> lstm_model
lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

# -------------> multi_window: 
multi_val_performance = {}
multi_performance = {}
OUT_STEPS = 72
multi_window = WindowGenerator(input_width=72,
                               label_width=OUT_STEPS,
                               train_df=train_df, val_df=val_df, test_df=test_df, pred_df=pred_df,
                               shift=OUT_STEPS)
multi_window.plot('SO2监测浓度(μg/m³)', './output/series_multi_window.pdf')
# ----------------------------------------------------------------

# -------------> base line: 
class MultiStepLastBaseline(tf.keras.Model):
  def call(self, inputs):
    return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

last_baseline = MultiStepLastBaseline()
last_baseline.compile(loss=tf.losses.MeanSquaredError(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance = {}
multi_performance = {}

multi_val_performance['Baseline'] = last_baseline.evaluate(multi_window.val)
multi_performance['Baseline'] = last_baseline.evaluate(multi_window.test, verbose=0)
multi_window.plot('SO2监测浓度(μg/m³)', './output/series_lastbaseline.pdf', model=last_baseline)

class RepeatBaseline(tf.keras.Model):
  def call(self, inputs):
    return inputs

repeat_baseline = RepeatBaseline()
repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                        metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val)
multi_performance['Repeat'] = repeat_baseline.evaluate(multi_window.test, verbose=0)
multi_window.plot('SO2监测浓度(μg/m³)', './output/series_repeatbaseline.pdf', model=repeat_baseline)
# ----------------------------------------------------------------


# -------------> linear
multi_linear_model = tf.keras.Sequential([
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_linear_model, multi_window)

#  multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
#  multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test, verbose=0)
multi_window.plot('SO2监测浓度(μg/m³)', './output/series_linear.pdf', model=multi_linear_model)
# ----------------------------------------------------------------

# -------------> RNN: LSTM
multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units].
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps*features].
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features].
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_lstm_model, multi_window)

multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
multi_window.plot('SO2监测浓度(μg/m³)', './output/series_lstm.pdf', model=multi_lstm_model)
# ----------------------------------------------------------------

# -------------> CNN
CONV_WIDTH = 3
multi_conv_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_conv_model, multi_window)

multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
multi_window.plot('SO2监测浓度(μg/m³)', './output/series_conv.pdf', model=multi_conv_model)
# ----------------------------------------------------------------

# -------------> feedback model: Autoregressive
feedback_model = FeedBack(units=32, out_steps=OUT_STEPS, num_features=num_features)
#  print('Output shape (batch, time, features): ', feedback_model(multi_window.example[0]).shape)
history = compile_and_fit(feedback_model, multi_window)
#  multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
#  multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)
multi_window.plot('SO2监测浓度(μg/m³)', './output/series_feedback.pdf', model=feedback_model)
# ----------------------------------------------------------------


# -------------> performance
x = np.arange(len(multi_performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = feedback_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]

plt.figure()
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=multi_performance.keys(),
           rotation=45)
plt.ylabel(f'MAE (average over all times and outputs)')
plt.legend()
plt.savefig('./output/performance_feedback.pdf')

for name, value in multi_performance.items():
  print(f'{name:8s}: {value[1]:0.4f}')
# ----------------------------------------------------------------


# -------------> plot prediction
prediction = feedback_model.predict(multi_window.test)
print(test_df.shape)
print(prediction.shape)

ts_out = pd.Series([row[0] for row in prediction[-1000:-1, 0:1, 0:1]])
ts_test = pd.Series(test_df['SO2监测浓度(μg/m³)'][-1000:-1].values)

plt.figure()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(ts_out)
ax2.plot(ts_test)
plt.savefig('./output/ts_out_test.pdf')
# ----------------------------------------------------------------


# -------------> get prediction
prediction = multi_conv_model.predict(multi_window.pred)
data_prediction = pd.DataFrame(prediction[-1, :, :], columns=df.columns)
data_prediction = data_prediction * pred_std + pred_mean
data_prediction.to_csv('./output/prediction_real_hour_72_A.csv')
print(train_std)

# ----------------------------------------------------------------


# -------------> complete
print('program complete')
