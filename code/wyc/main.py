#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd

data1_daily_real = pd.read_excel("/home/competition/math_modeling/questions/B/1.xlsx", sheet_name="监测点A逐日污染物浓度实测数据")

#  print(data1_daily_real.columns)

dict_range = pd.read_csv('./input/range.csv').to_dict(orient = 'list')

data = data1_daily_real.iloc[:1, ]

def calculate_AQI(data, dict_range):
    # ------------> init
    IAQI = pd.DataFrame()
    IAQIl = pd.DataFrame()
    IAQIh = pd.DataFrame()
    BPl = pd.DataFrame()
    BPh = pd.DataFrame()
    for key in dict_range:
        value = dict_range[key]

        print(value)
        sys.exit()

        # ------------> parts
        # IAQI low

        #  if key == 'SO2监测浓度(μg/m³)':
            #  print(value[0])
            #  print(value[1])
            #  print(data[key])
            #  sys.exit()

        IAQIl[key] = np.select(
            [
                data[key].between(value[0], value[1], inclusive='left'),
                data[key].between(value[1], value[2], inclusive='left'),
                data[key].between(value[2], value[3], inclusive='left'),
                data[key].between(value[3], value[4], inclusive='left'),
                data[key].between(value[4], value[5], inclusive='left'),
                data[key].between(value[5], value[6], inclusive='left'),
                data[key].between(value[6], value[7], inclusive='left'),
                data[key].between(value[7], np.inf, inclusive='left'),
            ],
            [0, 50, 100, 150, 200, 300, 400, 500],
            default = 'smaller_than_0'
        )
        # IAQI high
        IAQIh[key] = np.select(
            [
                data[key].between(value[0], value[1], inclusive='left'),
                data[key].between(value[1], value[2], inclusive='left'),
                data[key].between(value[2], value[3], inclusive='left'),
                data[key].between(value[3], value[4], inclusive='left'),
                data[key].between(value[4], value[5], inclusive='left'),
                data[key].between(value[5], value[6], inclusive='left'),
                data[key].between(value[6], value[7], inclusive='left'),
                data[key].between(value[7], np.inf, inclusive='left'),
            ],
            [50, 100, 150, 200, 300, 400, 500, np.nan],
            default = 'smaller_than_0'
        )
        # BP low
        BPl[key] = np.select(
            [
                data[key].between(value[0], value[1], inclusive='left'),
                data[key].between(value[1], value[2], inclusive='left'),
                data[key].between(value[2], value[3], inclusive='left'),
                data[key].between(value[3], value[4], inclusive='left'),
                data[key].between(value[4], value[5], inclusive='left'),
                data[key].between(value[5], value[6], inclusive='left'),
                data[key].between(value[6], value[7], inclusive='left'),
                data[key].between(value[7], np.inf, inclusive='left'),
            ],
            value,
            default = 'smaller_than_0'
            )
        # BP high
        BPh[key] = np.select(
            [
                data[key].between(value[0], value[1], inclusive='left'),
                data[key].between(value[1], value[2], inclusive='left'),
                data[key].between(value[2], value[3], inclusive='left'),
                data[key].between(value[3], value[4], inclusive='left'),
                data[key].between(value[4], value[5], inclusive='left'),
                data[key].between(value[5], value[6], inclusive='left'),
                data[key].between(value[6], value[7], inclusive='left'),
                data[key].between(value[7], np.inf, inclusive='left'),
            ],
            value[1:] + [np.nan],
            default = 'smaller_than_0'
        )
        # ------------> calculate
        #  IAQI[key] = (IAQIh[key] - IAQIl[key]) / (BPh[key] - BPl[key]) * (data[key] - BPl[key]) + IAQIl[key]
        print(value)
    print(data.values)
    print(IAQIh.values)
    print(IAQIl.values)
    print(BPh.values)
    print(BPl.values)
    #  print(IAQI)

calculate_AQI(data, dict_range)

