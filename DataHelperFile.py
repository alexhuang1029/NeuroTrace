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

def dataQuality(hdata):
  def data_quality_numeric(hdata):
    numeric_df = hdata(['air_time4','disp_index4','gmrt_in_air4','gmrt_on_paper4','max_x_extension4','max_y_extension4','mean_acc_in_air4','mean_acc_on_paper4','mean_gmrt4','mean_jerk_in_air4','mean_jerk_on_paper4','mean_speed_in_air4','mean_speed_on_paper4','num_of_pendown4','paper_time4','pressure_mean4','pressure_var4','total_time4']).describe(percentiles=[.25, .5, .75]).T # drop the ID column
    return numeric_df
