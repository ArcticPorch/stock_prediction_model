#!/usr/bin/env python
# coding: utf-8





import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense,LSTM,Dropout,GRU

get_ipython().run_line_magic('matplotlib', 'inline')






Data = pd.read_csv('/content/GOOG.csv',parse_dates =True)


