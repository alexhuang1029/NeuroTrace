## Import Prerequisites

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

## Getting hdata from csv

hdata = hdata = pd.read_csv('data.csv')

## Data Preprocessing

pd.set_option('display.float_format', lambda x: '%.5f' % x)
hdata['class'] = hdata['class'].map({'P':1,'H':0})

## Creating functions
def generate_column_names(columnNumber):
    base_column_names = ['air_time', 'disp_index', 'gmrt_in_air', 'gmrt_on_paper', 'max_x_extension',
                         'max_y_extension', 'mean_acc_in_air', 'mean_acc_on_paper', 'mean_gmrt',
                         'mean_jerk_in_air', 'mean_jerk_on_paper', 'mean_speed_in_air', 'mean_speed_on_paper',
                         'num_of_pendown', 'paper_time', 'pressure_mean', 'pressure_var', 'total_time']
    # Generate column names: appending columnNumber to each base column name
    column_name2 = [f'{base_column_name}{columnNumber}' for base_column_name in base_column_names]

    return column_name2

def summary(hdata):
  print('-' * 15)
  print(f'Total Number of Duplicated Data within DataFrame: {hdata[hdata.duplicated()].sum().sum()}')
  print('-' * 15)
  print(f"Total Number of Unique Data within DataFrame: {hdata.stack().nunique()}")

def dataQuality(columnNumber):
  base_column_names = ['air_time', 'disp_index', 'gmrt_in_air', 'gmrt_on_paper', 'max_x_extension',
                         'max_y_extension', 'mean_acc_in_air', 'mean_acc_on_paper', 'mean_gmrt',
                         'mean_jerk_in_air', 'mean_jerk_on_paper', 'mean_speed_in_air', 'mean_speed_on_paper',
                         'num_of_pendown', 'paper_time', 'pressure_mean', 'pressure_var', 'total_time']
    # Generate column names by appending columnNumber to each base column name
  column_names = [f'{base_column_name}{columnNumber}' for base_column_name in base_column_names]
  numeric_df = hdata[column_names].describe(percentiles=[.25, .5, .75]).T
  return numeric_df

def dataOutliers(hdata, columnNumber):
  base_column_names = ['air_time', 'disp_index', 'gmrt_in_air', 'gmrt_on_paper', 'max_x_extension',
                         'max_y_extension', 'mean_acc_in_air', 'mean_acc_on_paper', 'mean_gmrt',
                         'mean_jerk_in_air', 'mean_jerk_on_paper', 'mean_speed_in_air', 'mean_speed_on_paper',
                         'num_of_pendown', 'paper_time', 'pressure_mean', 'pressure_var', 'total_time']
    # Generate column names by appending columnNumber to each base column name
  column_names = [f'{base_column_name}{columnNumber}' for base_column_name in base_column_names]
  fig, axes = plt.subplots(nrows=round((len(base_column_names))), ncols=1, figsize=(20, 100))
  axes = axes.flatten()

    # Plot boxplots for each column
  for i, column in enumerate(column_names):
        sns.boxplot(x=hdata[column], ax=axes[i])
        axes[i].set_title(f'Outliers boxplot of {column}')

  plt.show()

def scatterPlot(hdata, columnNumber):
  base_column_names = ['air_time', 'disp_index', 'gmrt_in_air', 'gmrt_on_paper', 'max_x_extension',
                         'max_y_extension', 'mean_acc_in_air', 'mean_acc_on_paper', 'mean_gmrt',
                         'mean_jerk_in_air', 'mean_jerk_on_paper', 'mean_speed_in_air', 'mean_speed_on_paper',
                         'num_of_pendown', 'paper_time', 'pressure_mean', 'pressure_var', 'total_time']
    # Generate column names by appending columnNumber to each base column name
  column_names = [f'{base_column_name}{columnNumber}' for base_column_name in base_column_names]
  fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
  axes = axes.flatten()

  # Scatterplot plot
  sns.scatterplot(data=hdata, x='gmrt_on_paper2', y='total_time2')
  axes[1].set_title(f'Scatterplot of total time vs Mean Relative Tremor')
  plt.show()
