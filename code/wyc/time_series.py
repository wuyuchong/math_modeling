#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------> imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#  import tensorflow as tf
from src.calculate_AQI import calculate_AQI

mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# -------------> data wrangling
data1_hourly_real = pd.read_excel("/home/competition/math_modeling/questions/B/1.xlsx", sheet_name="监测点A逐小时污染物浓度与气象实测数据", na_values='—')
#  data1_hourly_report = pd.read_excel("/home/competition/math_modeling/questions/B/1.xlsx", sheet_name="监测点A逐小时污染物浓度与气象一次预报数据", na_values='-')

print(data1_hourly_real.info())
print(data1_hourly_real.columns)

df = data1_hourly_real.copy()
date_time = pd.to_datetime(df.pop('监测时间'), format='%d-%m-%Y %H:%M:%S')
df.pop('地点')
print(df.describe().transpose())

# -------------> feature plot: long term
plot_cols = ['SO2监测浓度(μg/m³)', '气压(MBar)']
plot_features = df[plot_cols]
plot_features.index = date_time
plot_features.plot(subplots=True)
plt.savefig('./output/feature_long_term.png')

# -------------> feature plot: short term
plot_features = df[plot_cols][:48]
plot_features.index = date_time[:48]
plt.savefig('./output/feature_short_term.png')

# -------------> exception processing: waiting

# -------------> wind visulization: waiting
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
#  plt.hist2d(df['Wx'], df['Wy'], bins=(50, 50), vmax=400)
#  plt.colorbar()
#  plt.xlabel('Wind X [m/s]')
#  plt.ylabel('Wind Y [m/s]')
#  ax = plt.gca()
#  ax.axis('tight')

# -------------> time transform
timestamp_s = date_time.map(pd.Timestamp.timestamp)
day = 24*60*60
year = (365.2425)*day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

print('programe complete')
