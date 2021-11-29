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
plt.savefig('./output/feature_long_term.png')

# -------------> feature plot: short term
plot_features = df[plot_cols][:48]
plot_features.index = date_time[:48]
plt.figure()
plt.savefig('./output/feature_short_term.png')

# -------------> exception processing: waiting

# -------------> wind visulization: waiting
#  plt.figure()
#  plt.hist2d(df['风向(°)'], df['风速(m/s)'])
#  plt.colorbar()
#  plt.savefig('./output/wind.png')
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
plt.savefig('./output/time_transform.png')


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
plt.savefig('./output/frequency.png')
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
plt.savefig('./output/features_distribution.png')
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

# ----------------------------------------------------------------

print('programe complete')
