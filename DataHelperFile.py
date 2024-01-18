## Import Prerequisites

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

## Getting hdata from csv

hdata = hdata = pd.read_csv('data.csv')

## Data Preprocessing

pd.set_option('display.float_format', lambda x: '%.7f' % x)
hdata['class'] = hdata['class'].map({'P':1,'H':0})

## Creating functions

def summary(hdata):
  print('-' * 15)
  print(f'Total Number of Duplicated Data within DataFrame: {hdata[hdata.duplicated()].sum().sum()}')
  print('-' * 15)
  print(f"Total Number of Unique Data within DataFrame: {hdata.stack().nunique()}")
## hello!!
