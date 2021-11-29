#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from plotnine import ggplot, geom_point, aes, stat_smooth, xlim, ylim, facet_grid
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix
print(tf.__version__)

dat = pd.read_csv('data/main/data_for_model.csv', dtype = {'code': str})

#  dat = dat.loc[~ np.isnan(dat['diff']), ]
#  print(dat.year1.describe())
#  print(dat.year2.describe())
#  sys.exit()

dat = dat.drop(['isST', 'year2', 'unit', 'market'], axis=1)
code_name = pd.read_csv('data/main/data_for_model.csv')

effect_dataset = dat.query('year1 == 2019').dropna()
effect_code = effect_dataset.pop('code')
effect_labels = effect_dataset.pop('diff')

predict_dataset = dat.query('year1 == 2020').drop('diff', axis=1).dropna()
predict_code = predict_dataset.pop('code')

dataset = dat.dropna().drop(['code'], axis=1)
backtest_dataset = dataset.query('year1 == 2018')
dataset = dataset.query('year1 < 2018')

train_dataset = dataset.sample(frac=0.8)
validation_dataset = dataset.drop(train_dataset.index).sample(frac = 0.5)
test_dataset = dataset.drop(train_dataset.index).drop(validation_dataset.index)

train_labels = train_dataset.pop('diff')
validation_labels = validation_dataset.pop('diff')
test_labels = test_dataset.pop('diff')
backtest_labels = backtest_dataset.pop('diff')

print(train_dataset.shape)
print(validation_dataset.shape)
print(test_dataset.shape)
print(backtest_dataset.shape)
print(effect_dataset.shape)
print(predict_dataset.shape)

# outlier preprocessing
def outlier(dataset):
    dat = dataset.copy()
    for column in dat.columns:
        low = dat[column].describe().loc['25%']
        high = dat[column].describe().loc['75%']
        dat.loc[dat[column] < low, column] = low
        dat.loc[dat[column] > high, column] = high
    return dat

train_dataset = outlier(train_dataset)
validation_dataset = outlier(validation_dataset)
test_dataset = outlier(test_dataset)
backtest_dataset = outlier(backtest_dataset)
effect_dataset = outlier(effect_dataset)
predict_dataset = outlier(predict_dataset)

train_stats = train_dataset.describe()
train_stats = train_stats.transpose()

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_validation_data = norm(validation_dataset)
normed_test_data = norm(test_dataset)
normed_backtest_data = norm(backtest_dataset)
normed_effect_data = norm(effect_dataset)
normed_predict_data = norm(predict_dataset)

print(effect_labels.describe())
pd.concat([train_labels.describe(), validation_labels.describe(), test_labels.describe(), backtest_labels.describe()], axis=1).to_csv('doc/outcome/train/标签的分布.csv')

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
plt.figure(figsize = (6, 3))
plt.plot(step/STEPS_PER_EPOCH, lr)
plt.ylim([0,max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.savefig('doc/outcome/train/学习率随迭代次数变化曲线.pdf')

model = build_model()
model.summary()

# 通过为每个完成的时期打印一个点来显示训练进度
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')


#  hist = pd.DataFrame(history.history)
#  hist['epoch'] = history.epoch
#  print(hist.tail())

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure(figsize = (10, 4))
  
  plt.subplot(1, 2, 1)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,1])
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,1])
  plt.legend()
  plt.savefig('doc/outcome/train/随迭代次数增加损失的变化.pdf')

model = build_model()
EPOCHS = 1000

#  patience 值用来检查改进 epochs 的数量
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_data = (normed_validation_data, validation_labels), verbose=0, callbacks=[PrintDot()])
plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

# ------- test

test_predictions = model.predict(normed_test_data).flatten()
table = pd.DataFrame({'actual': test_labels, 'prediction': test_predictions})
table1 = table.copy()

#  plt.scatter(test_labels, test_predictions)
#  plt.xlabel('True Values')
#  plt.ylabel('Predictions')
#  plt.axis('equal')
#  plt.axis('square')
#  plt.xlim([-1,1])
#  plt.ylim([-1,1])
#  plt.plot([-100, 100], [-100, 100])
#  plt.show()

error = test_predictions - test_labels
plt.figure(figsize = (10, 4))
plt.subplot(1, 2, 1)
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error")
plt.ylabel("Count")

#  (ggplot(table, aes('actual', 'prediction'))
#  + geom_point(alpha=0.1)
#  + stat_smooth(method='glm')
#  ).save('doc/outcome/train/DNN在测试集上的表现.pdf')

table.loc[table['actual'] >= 0, 'actual'] = 1
table.loc[table['actual'] <  0, 'actual'] = 0
table.loc[table['prediction'] >= 0, 'prediction'] = 1
table.loc[table['prediction'] <  0, 'prediction'] = 0

tn, fp, fn, tp = confusion_matrix(table['actual'], table['prediction']).ravel()

print('Test Correlation Coefficient: ', np.corrcoef(table.actual, table.prediction))

print(confusion_matrix(table['actual'], table['prediction']))
print('Accuracy:', (tp + tn) / (tn + fp + fn + tp))

# backtest -------

backtest_predictions = model.predict(normed_backtest_data).flatten()
table = pd.DataFrame({'actual': backtest_labels, 'prediction': backtest_predictions})
table2 = table.copy()

#  plt.scatter(backtest_labels, backtest_predictions)
#  plt.xlabel('True Values')
#  plt.ylabel('Predictions')
#  plt.axis('equal')
#  plt.axis('square')
#  plt.xlim([-1,1])
#  plt.ylim([-1,1])
#  plt.plot([-100, 100], [-100, 100])
#  plt.show()

error = backtest_predictions - backtest_labels
plt.subplot(1, 2, 2)
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error")
plt.ylabel("Count")
plt.savefig('doc/outcome/train/DNN在测试集和回测集上的残差分布.pdf')

table1 = table1.assign(dataset='Test Dataset')
table2 = table2.assign(dataset='BackTest Dataset')
table_combine = pd.concat([table1, table2], axis=0)
table_combine['dataset'] = table_combine['dataset'].astype('category')
table_combine['dataset'].cat.reorder_categories(['Test Dataset', 'BackTest Dataset'], inplace=True, ordered=True)
(ggplot(table_combine, aes('actual', 'prediction'))
+ geom_point(alpha=0.1)
+ stat_smooth(method='glm')
+ facet_grid('. ~ dataset')
).save('doc/outcome/train/DNN在测试集和回测集上的表现.pdf', height=4, width=10)

print('Backtest Correlation Coefficient: ', np.corrcoef(table.actual, table.prediction))

table.loc[table['actual'] >= 0, 'actual'] = 1
table.loc[table['actual'] <  0, 'actual'] = 0
table.loc[table['prediction'] >= -0.4, 'prediction'] = 1
table.loc[table['prediction'] <  -0.4, 'prediction'] = 0

tn, fp, fn, tp = confusion_matrix(table['actual'], table['prediction']).ravel()

print(confusion_matrix(table['actual'], table['prediction']))
print('Accuracy:', (tp + tn) / (tn + fp + fn + tp))

# --------- effect

effect_predictions = model.predict(normed_effect_data).flatten()
table = pd.DataFrame({'code': effect_code, 'actual': effect_labels, 'prediction': effect_predictions})
print(table)
table.to_csv('doc/outcome/train/effect_predictions.csv', index=False)
print(table.actual.mean())

print('Effect Correlation Coefficient: ', np.corrcoef(table.actual, table.prediction))

# --------- prediction

predict_predictions = model.predict(normed_predict_data).flatten()
table = pd.DataFrame({'code': predict_code, 'prediction': predict_predictions})
print(table)
table.to_csv('doc/outcome/train/predict_predictions.csv', index=False)


