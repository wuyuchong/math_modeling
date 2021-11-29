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

# -------------> configuration
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# -------------> data wrangling
data1_hourly_real = pd.read_excel("/home/competition/math_modeling/questions/B/1.xlsx", sheet_name="监测点A逐小时污染物浓度与气象实测数据", na_values='—')
#  data1_hourly_report = pd.read_excel("/home/competition/math_modeling/questions/B/1.xlsx", sheet_name="监测点A逐小时污染物浓度与气象一次预报数据", na_values='-')
#  print(data1_hourly_real.info())
df = data1_hourly_real.copy()
date_time = pd.to_datetime(df.pop('监测时间'), format='%Y-%m-%d %H:%M:%S')
df.pop('地点')
#  print(df.describe().transpose())

# -------------> feature plot: long term
plot_cols = ['SO2监测浓度(μg/m³)', '气压(MBar)']
plot_features = df[plot_cols]
plot_features.index = date_time
plt.figure()
plot_features.plot(subplots=True)
plt.savefig('./output/feature_long_term.pdf')

# -------------> feature plot: short term
plot_features = df[plot_cols][:48]
plot_features.index = date_time[:48]
plt.figure()
plt.savefig('./output/feature_short_term.pdf')

# -------------> exception processing: waiting
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp_mean = IterativeImputer(random_state=0)
imp_mean.fit(df.values)
imp = imp_mean.transform(df.values)
df_imp = pd.DataFrame(imp, columns=df.columns)
#  print(df.head())
#  print(df_imp.head())
#  print(df.isna().any())
#  print(df_imp.isna().any())
df = df_imp

# -------------> wind visulization: waiting
#  plt.figure()
#  plt.hist2d(df['风向(°)'], df['风速(m/s)'])
#  plt.colorbar()
#  plt.savefig('./output/wind.pdf')
#  print(df['风向(°)'].describe())
#  print(df['风速(m/s)'].describe())

# -------------> wind direction transform: waiting
wv = df.pop('风速(m/s)')
wd_rad = df.pop('风向(°)')*np.pi / 180
df['Wx'] = wv*np.cos(wd_rad)
df['Wy'] = wv*np.sin(wd_rad)
#  plt.figure()
#  plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
#  plt.colorbar()
#  plt.xlabel('Wind X [m/s]')
#  plt.ylabel('Wind Y [m/s]')
#  ax = plt.gca()
#  ax.axis('tight')

# -------------> time transform: waiting for half day
timestamp_s = date_time.map(pd.Timestamp.timestamp)
day = 24*60*60
year = (365.2425)*day
df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
plt.figure()
plt.plot(np.array(df['Day sin'])[:25])
plt.plot(np.array(df['Day cos'])[:25])
plt.xlabel('Time [h]')
plt.title('Time of day signal')
plt.savefig('./output/time_transform.pdf')


# -------------> frequency
#  fft = tf.signal.rfft(df['SO2监测浓度(μg/m³)'].dropna())
fft = tf.signal.rfft(df['O3监测浓度(μg/m³)'].dropna())
f_per_dataset = np.arange(0, len(fft))
#  n_samples_h = len(df['SO2监测浓度(μg/m³)'])
n_samples_h = len(df['O3监测浓度(μg/m³)'])
hours_per_year = 24*365.2524
years_per_dataset = n_samples_h/(hours_per_year)
f_per_year = f_per_dataset/years_per_dataset
plt.figure()
plt.step(f_per_year, np.abs(fft))
plt.xscale('log')
#  plt.ylim(0, 400000)
plt.xlim([0.1, max(plt.xlim())])
plt.xticks([1, 365.2524, 731], labels=['1/Year', '1/day', '12/hour'])
#  plt.xticks([1, 30.4377, 91.3, 365.2524, 8766.06], labels=['1/year', '1/season', '1/month' '1/day', '1/hour'])
plt.xlabel('Frequency (log scale)')
plt.savefig('./output/frequency.pdf')
# ----------------------------------------------------------------

# -------------> dataset split
column_indices = {name: i for i, name in enumerate(df.columns)}
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]
num_features = df.shape[1]
# ----------------------------------------------------------------


# -------------> normalization: waiting for moving average
train_mean = train_df.mean()
train_std = train_df.std()
train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

plt.figure()
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
ax.set_xticklabels(df.keys(), rotation=90)
plt.savefig('./output/features_distribution.pdf')
# ----------------------------------------------------------------


# -------------> WindowGenerator
w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                     train_df=train_df, val_df=val_df, test_df=test_df,
                     label_columns=['SO2监测浓度(μg/m³)'])
# Stack three slices, the length of the total window.
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100+w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size])])
example_inputs, example_labels = w2.split_window(example_window)
#  print('All shapes are: (batch, time, features)')
#  print(f'Window shape: {example_window.shape}')
#  print(f'Inputs shape: {example_inputs.shape}')
#  print(f'Labels shape: {example_labels.shape}')
#  repr(w2)
#  print(w2.train.element_spec)
#  for example_inputs, example_labels in w2.train.take(1):
  #  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  #  print(f'Labels shape (batch, time, features): {example_labels.shape}')
#
# ----------------------------------------------------------------


# -------------> single_step_window
single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=['SO2监测浓度(μg/m³)'])
#  print(single_step_window)
for example_inputs, example_labels in single_step_window.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')
baseline = Baseline(label_index=column_indices['SO2监测浓度(μg/m³)'])
baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)
print(val_performance)
print(performance)

wide_window = WindowGenerator(
    input_width=24, label_width=24, shift=1,
    train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=['SO2监测浓度(μg/m³)'])
#  print(wide_window)
#  print('Input shape:', wide_window.example[0].shape)
#  print('Output shape:', baseline(wide_window.example[0]).shape)
wide_window.plot('SO2监测浓度(μg/m³)', './output/series_base_line.pdf', model=baseline)
# ----------------------------------------------------------------


# -------------> Linear model
linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])
#  print('Input shape:', single_step_window.example[0].shape)
#  print('Output shape:', linear(single_step_window.example[0]).shape)
MAX_EPOCHS = 20
history = compile_and_fit(linear, single_step_window, MAX_EPOCHS)
val_performance['Linear'] = linear.evaluate(single_step_window.val)
performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)

#  print('Input shape:', wide_window.example[0].shape)
#  print('Output shape:', baseline(wide_window.example[0]).shape)
wide_window.plot('SO2监测浓度(μg/m³)', './output/series_linear.pdf', model=linear)

plt.figure()
plt.bar(x = range(len(train_df.columns)),
        height=linear.layers[0].kernel[:,0].numpy())
axis = plt.gca()
axis.set_xticks(range(len(train_df.columns)))
axis.set_xticklabels(train_df.columns, rotation=90)
plt.savefig('./output/importence_linear.pdf')
# ----------------------------------------------------------------


# -------------> Dense
dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
history = compile_and_fit(dense, single_step_window, MAX_EPOCHS)
val_performance['Dense'] = dense.evaluate(single_step_window.val)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)
print(performance)
# ----------------------------------------------------------------


# -------------> Multi-step dense
CONV_WIDTH = 3
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=['SO2监测浓度(μg/m³)'])
conv_window.plot('SO2监测浓度(μg/m³)', './output/conv_window.pdf')
multi_step_dense = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', multi_step_dense(conv_window.example[0]).shape)

out = multi_step_dense.predict(conv_window.test)
print(test_df.shape)
print(out.shape)

ts_out = pd.Series([row[0] for row in out[-100:-1, :, 0]])
ts_test = pd.Series(test_df['SO2监测浓度(μg/m³)'][-100:-1].values)

plt.figure()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(ts_out)
ax2.plot(ts_test)
plt.savefig('./output/ts_out_test.pdf')


print('program complete')
