#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 首要污染物可能出现并列

import numpy as np
import pandas as pd

def calculate_AQI(raw, dict_range):
    # ------------> init
    data = raw.copy()
    IAQI = pd.DataFrame()
    IAQIl = pd.DataFrame()
    IAQIh = pd.DataFrame()
    BPl = pd.DataFrame()
    BPh = pd.DataFrame()
    for key in dict_range:
        value = dict_range[key]
        # ------------> parts
        # IAQI low
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
            default = np.nan
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
            default = np.nan
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
            default = np.nan
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
            default = np.nan
        )
    # ------------> calculate
    IAQIh = IAQIh.apply(pd.to_numeric)
    IAQIl = IAQIl.apply(pd.to_numeric)
    BPh = BPh.apply(pd.to_numeric)
    BPl = BPl.apply(pd.to_numeric)
    for key in dict_range:
        IAQI[key] = (IAQIh[key] - IAQIl[key]) / (BPh[key] - BPl[key]) * (data[key] - BPl[key]) + IAQIl[key]
    data['IAQ'] = np.ceil(IAQI.max(axis=1))
    # 取最大值时有一项为 NA 则整体为 NA
    data.loc[IAQI.isna().any(axis=1), 'IAQ'] = np.nan
    # 小于等于 50 无首要污染物
    data['element'] = np.nan
    data.loc[data['IAQ'] > 50, 'element'] = IAQI.idxmax(axis=1)
    # ------------> output
    print('已经在原数据集的基础上添加两列，注意：\n 空气质量爆表或观测缺失都会导致 NA \n 取最大值时有一项为 NA 则整体为 NA \n 小于等于 50 无首要污染物')
    return data

