#!/usr/bin/env python
# -*- coding: utf-8 *-

import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from plotnine import ggplot, geom_point, aes, stat_smooth, xlim, ylim, facet_grid, labs, theme_light
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from src.calculate_AQI import calculate_AQI
print(tf.__version__)

# -------------> configuration
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

A = pd.read_excel("/home/competition/math_modeling/questions/B/1.xlsx", sheet_name="监测点A逐小时污染物浓度与气象实测数据", na_values='—')
A1 = pd.read_excel("/home/competition/math_modeling/questions/B/3.xlsx", sheet_name="监测点A1逐小时污染物浓度与气象实测数据", na_values='—')
A2 = pd.read_excel("/home/competition/math_modeling/questions/B/3.xlsx", sheet_name="监测点A2逐小时污染物浓度与气象实测数据", na_values='—')
A3 = pd.read_excel("/home/competition/math_modeling/questions/B/3.xlsx", sheet_name="监测点A3逐小时污染物浓度与气象实测数据", na_values='—')

def process(data, s):
    # wind
    if s == '0':
        wv = data.pop('风速(m/s)')
    else:
        wv = data.pop('近地风速(m/s)')
    wd_rad = data.pop('风向(°)')*np.pi / 180
    data['Wx'] = wv*np.cos(wd_rad)
    data['Wy'] = wv*np.sin(wd_rad)
    # datetime
    date_time = pd.to_datetime(data['监测时间'], format='%Y-%m-%d %H:%M:%S')
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    day = 24*60*60
    year = (365.2425)*day
    data['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    data['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    data['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    data['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    # col_names
    data.pop('地点')
    if s != '0':
        col_names = ['监测时间'] + [(name + '_' + s) for name in data.columns if name != '监测时间']
        data.columns = col_names
    return data

# -------------> AQIh
dict_range = pd.read_csv('./input/range_hour.csv').to_dict(orient = 'list')
A = calculate_AQI(A, dict_range)
A.pop('element')

# -------------> process and merge
A = process(A, '0')
A1 = process(A1, '1')
A2 = process(A2, '2')
A3 = process(A3, '3')
df = pd.merge(A, A1, on = '监测时间')
df = pd.merge(df, A2, on = '监测时间')
df = pd.merge(df, A3, on = '监测时间')
df.pop('监测时间')
print(df.shape)

# -------------> null processing
imp_mean = IterativeImputer(random_state=0)
imp_mean.fit(df.values)
imp = imp_mean.transform(df.values)
df_imp = pd.DataFrame(imp, columns=df.columns)
df = df_imp

# -------------> dataset split
n = len(df)
dataset = df
train_dataset = df[0:int(n*0.7)]
validation_dataset = df[int(n*0.7):int(n*0.9)]
test_dataset = df[int(n*0.9):]
num_features = df.shape[1]

train_labels = train_dataset.pop('AQI')
validation_labels = validation_dataset.pop('AQI')
test_labels = test_dataset.pop('AQI')

# -------------> norm
train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_validation_data = norm(validation_dataset)
normed_test_data = norm(test_dataset)

# -------------> model
STEPS_PER_EPOCH = 20
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.00001,
  decay_steps=STEPS_PER_EPOCH*10,
  decay_rate=0.01,
  staircase=False)

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=[len(train_dataset.keys())]),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(1)
  ])
  optimizer = tf.keras.optimizers.RMSprop(lr_schedule)
  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

step = np.linspace(0,100000)
lr = lr_schedule(step)
plt.figure(figsize = (6, 4))
plt.plot(step/STEPS_PER_EPOCH, lr)
plt.ylim([0,max(plt.ylim())])
plt.xlabel('迭代次数')
plt.ylabel('学习率')
plt.savefig('./output/学习率随迭代次数变化曲线.pdf')

model = build_model()
model.summary()

# 通过为每个完成的时期打印一个点来显示训练进度
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize = (4, 4))
  plt.xlabel('迭代次数')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  #  plt.ylim([0,1])
  plt.legend()
  plt.savefig('./output/train/随迭代次数增加损失的变化.pdf')

model = build_model()
EPOCHS = 1000

#  patience 值用来检查改进 epochs 的数量
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_data = (normed_validation_data, validation_labels), verbose=0, callbacks=[PrintDot()])
plot_history(history)
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

# -------------> test
test_predictions = model.predict(normed_test_data).flatten()
table = pd.DataFrame({'actual': test_labels, 'prediction': test_predictions})

error = test_predictions - test_labels
plt.figure(figsize = (5, 4))
plt.hist(error, bins = 25)
plt.xlabel("预测误差")
plt.ylabel("量")
plt.savefig('./output/train/DNN 在测试集上的残差分布.pdf', height=4, width=4)

(ggplot(table, aes('actual', 'prediction'))
+ geom_point(alpha=0.1)
+ stat_smooth(method='glm')
+ labs(x = 'AQI/h 真实值', y = 'AQI/h 二次预测值')
+ theme_light(base_family = "SimHei")
).save('./output/train/DNN 在测试集上的表现.pdf', height=4, width=4)
