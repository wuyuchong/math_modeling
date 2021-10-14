#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
from src.calculate_AQI import calculate_AQI

data1_daily_real = pd.read_excel("/home/competition/math_modeling/questions/B/1.xlsx", sheet_name="监测点A逐日污染物浓度实测数据")
dict_range = pd.read_csv('./input/range.csv').to_dict(orient = 'list')

data = data1_daily_real.copy()
calculate_AQI(data, dict_range)
print(data)

