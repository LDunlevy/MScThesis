# Scenario Building
# This script is used to build the scenarios for San Clemente Island
# The scenarios are built using the following steps:
#%% 
#Import the necessary libraries
import pandas as pd
import numpy as np
#import geopandas as gpd
import matplotlib.pyplot as plt
import os
import random
import math
import time
import datetime
import copy
from scipy.ndimage import gaussian_filter1d
import scipy.signal as signal
from scipy.signal import savgol_filter
#make plot default more academic (bold axis, etc)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["axes.linewidth"] = 2
plt.rcParams["xtick.major.width"] = 2
plt.rcParams["ytick.major.width"] = 2
plt.rcParams["xtick.minor.width"] = 1.5
plt.rcParams["ytick.minor.width"] = 1.5
plt.rcParams["xtick.major.size"] = 5
plt.rcParams["ytick.major.size"] = 5
plt.rcParams["xtick.minor.size"] = 3
plt.rcParams["ytick.minor.size"] = 3
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["legend.title_fontsize"] = 12
#import seaborn as sns
#%%
#Format raw data
# Daily load 
# Load the daily load data
daily_load_1999 = pd.read_csv('digitized/daily1999.csv')
monthlyKWh = pd.read_csv('digitized/monthlyKWhannum.csv')
#convert monthlyKWh to datetime
monthlyKWh['Date'] = pd.to_datetime(monthlyKWh['date'],format='%Y/%m')
#delete the date column
monthlyKWh.drop(columns=['date'], inplace=True)
# create a new column called std that takes the mean of the difference between stdup and mean and stddown and mean for each year
daily_load_1999['std'] = (daily_load_1999['stdup'] - daily_load_1999['mean'] + daily_load_1999['mean'] - daily_load_1999['stddown'])/2


# %%
# Create sim data infrastructure, used to generate simulated data
# Create date range for simulated data
date_range = pd.date_range(start='1/1/1999', end='1/1/2000', freq='h')
date_range = date_range[:-1]
#offset date range by 30 min
date_range = date_range + pd.DateOffset(minutes=30)
# generate simlated data placeholder dataframe with the first column as the date
sim_data_inf = pd.DataFrame(date_range, columns=['Date'])
# head to check
#sim_data_inf.head()
# attach mean and std to respective times in the simulated data
# create a new column for the mean and std and max and min
sim_data_inf['std'] = 0
sim_data_inf['mean'] = 0
sim_data_inf['max'] = 0
sim_data_inf['min'] = 0
#convert the columns to float

sim_data_inf['std'] = sim_data_inf['std'].astype(float)
sim_data_inf['mean'] = sim_data_inf['mean'].astype(float)
sim_data_inf['max'] = sim_data_inf['max'].astype(float)
sim_data_inf['min'] = sim_data_inf['min'].astype(float)
# loop through the simulated data and attach the mean and std to the respective times
for i in range(len(sim_data_inf)):
    sim_data_inf.loc[i, 'mean'] = daily_load_1999['mean'][sim_data_inf['Date'][i].hour]
    sim_data_inf.loc[i, 'std'] = daily_load_1999['std'][sim_data_inf['Date'][i].hour]
    sim_data_inf.loc[i, 'max'] = daily_load_1999['max'][sim_data_inf['Date'][i].hour]
    sim_data_inf.loc[i, 'min'] = daily_load_1999['min'][sim_data_inf['Date'][i].hour]

# Convert the 'Date' column to datetime if it's not already
sim_data_inf['Date'] = pd.to_datetime(sim_data_inf['Date'])

# Set the 'Date' column as the index
sim_data_inf.set_index('Date', inplace=True)
# head to check
#sim_data_inf.head()
#let me see more
#sim_data_inf.tail()
#print the sim_data_inf means I want to see all not just head and tail
print(daily_load_1999)


#%%
# Begin genrating simulated data using normal curves from daily load
# another data frame that uses the mean and std from sim_data_inf to generate kWhs for every hour
sim_data = pd.DataFrame(date_range, columns=['Date'])
sim_data['kWh'] = 0
sim_data['kWh'] = sim_data['kWh'].astype(float)

#loop through the sim_data and generate a vaue using a random distribution with mean and std from the same time in sim_data_inf
for i in range(len(sim_data)):
    sim_data.loc[i, 'kWh'] = np.random.normal(sim_data_inf['mean'].iloc[i], sim_data_inf['std'].iloc[i], 1)# head to check
#sim_data.head()
# Convert the 'Date' column to datetime if it's not already
sim_data['Date'] = pd.to_datetime(sim_data['Date'])

# Set the 'Date' column as the index
sim_data.set_index('Date', inplace=True)

def weighted_moving_average(data, window_size):
    weights = np.arange(1, window_size + 1)
    return np.convolve(data, weights/weights.sum(), mode='valid')
#sim_data['smoothed'] = gaussian_filter1d(sim_data['kWh'], sigma=1.5)
#sim_data['smoothed'] = signal.medfilt(sim_data['kWh'], kernel_size=3)
#sim_data['smoothed'] = weighted_moving_average(sim_data['kWh'], window_size=5)
sim_data['smoothed'] = savgol_filter(sim_data['kWh'], window_length=7, polyorder=2)
# %%
# Set a scaling factor to match monthly totals
# Group by month and calculate the total
monthly_totals = sim_data.resample('ME').sum()
monthly_totals.head()
#set monthlyKWh and monthly_totals to the same index
monthlyKWh.index = monthly_totals.index
#delete date column
monthlyKWh.drop(columns=['Date'], inplace=True)
#set type to float
monthlyKWh['KWh'] = monthlyKWh['KWh'].astype(float)
#divide them to get scaling factor
scaling_factoro = monthlyKWh['KWh']/monthly_totals['kWh']
scaling_factors = monthlyKWh['KWh']/monthly_totals['smoothed']


# %%
# Apply the scaling factor to the simulated data
#in sim_data_inf create a new column called scaling factor and set it equal to the scaling factor that belongs to that month
sim_data_inf['ScalingFactor'] = 0

# Resample 'ScalingFactor' to match 'sim_data_inf' index
scaling_factoro = scaling_factoro.reindex(sim_data_inf.index, method='ffill')
scaling_factoro = scaling_factoro.bfill()

scaling_factors = scaling_factoro.reindex(sim_data_inf.index, method='ffill')
scaling_factors = scaling_factoro.bfill()

# Attach the scaling factor to sim_data_inf
sim_data_inf['ScalingFactoro'] = scaling_factoro
sim_data_inf['ScalingFactors'] = scaling_factors
#multiply the kWh in sim_data by the scaling factor
sim_data['kWh'] = sim_data['kWh'] * sim_data_inf['ScalingFactoro']
sim_data['smoothed'] = sim_data['smoothed'] * sim_data_inf['ScalingFactors']
# check the new monthly totals
monthly_totals = sim_data.resample('ME').sum()
monthly_totals.head()


# %%
# Plot all data
plt.figure(figsize=(12, 6))
plt.plot(sim_data['kWh'], label='Simulated Data')
plt.plot(sim_data['smoothed'], label='Smoothed Data')
#plt.plot(monthlyKWh['KWh'], label='Actual Data')
plt.xlabel('Date')
plt.ylabel('kWh')
plt.title('Simulated Data')
#%%
# limit sim_data to the max and min values in sim_data_inf
sim_data['kWh'] = sim_data['kWh'].clip(sim_data_inf['min'], sim_data_inf['max'])
sim_data['smoothed'] = sim_data['smoothed'].clip(sim_data_inf['min'], sim_data_inf['max'])
#%% 
#convert the index to datetime
sim_data.index = pd.to_datetime(sim_data.index, format='%m/%d/%Y %H:%M')
#head to check
sim_data.head()
#export sim_data to a csv
sim_data.to_csv('sim_data.csv')
# %%
# Plot all data
plt.figure(figsize=(12, 6))
plt.plot(sim_data['kWh'], label='Simulated Data')
plt.plot(sim_data['smoothed'], label='Smoothed Data')
#plt.plot(monthlyKWh['KWh'], label='Actual Data')
plt.xlabel('Date')
plt.ylabel('kWh')
plt.title('Simulated Data')

#%% create kWhs to make graph copying the initial graph from the snippet

# Extract the time from the 'Date' column
sim_data['Time'] = sim_data.index.time

# Group by time and calculate the mean, std, max, and min kWhs
mean_kWhs = sim_data.groupby('Time').mean()
std_kWhs = sim_data.groupby('Time').std()
max_kWhs = sim_data.groupby('Time').max()
min_kWhs = sim_data.groupby('Time').min()


# %%
#plot all of the above statistical kWh on the same plot in a daily load curve
#plot a std devation above and below the mean
#give each their own symbol
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

# Convert the index to a DatetimeIndex
mean_kWhs.index = pd.to_datetime(mean_kWhs.index, format='%H:%M:%S')
std_kWhs.index = pd.to_datetime(std_kWhs.index, format='%H:%M:%S')
max_kWhs.index = pd.to_datetime(max_kWhs.index, format='%H:%M:%S')
min_kWhs.index = pd.to_datetime(min_kWhs.index, format='%H:%M:%S')
daily_load_1999.index = pd.to_datetime(mean_kWhs.index, format='%H:%M:%S')

# Create a date locator and formatter
hours = mdates.HourLocator(interval=1)
h_fmt = mdates.DateFormatter('%H:%M:%S')

# Plot the data
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(mean_kWhs['kWh'], label='Mean', color='black')
ax.fill_between(mean_kWhs.index, mean_kWhs['kWh'] + std_kWhs['kWh'], mean_kWhs['kWh'] - std_kWhs['kWh'], color='gray', alpha=0.5, label='Std Dev')
ax.scatter(max_kWhs.index, max_kWhs['kWh'], color='red', label='Max')
ax.scatter(min_kWhs.index, min_kWhs['kWh'], color='blue', label='Min')
ax.xaxis.set_major_locator(hours)
ax.xaxis.set_major_formatter(h_fmt)
plt.xticks(rotation=45)
plt.ylim(0, 1600)
plt.xlabel('Time')
plt.ylabel('kWh')
plt.title('Daily Load Curve: No Smoothing')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(mean_kWhs['smoothed'], label='Mean', color='black')
ax.fill_between(mean_kWhs.index, mean_kWhs['smoothed'] + std_kWhs['kWh'], mean_kWhs['smoothed'] - std_kWhs['kWh'], color='gray', alpha=0.5, label='Std Dev')
ax.scatter(max_kWhs.index, max_kWhs['smoothed'], color='red', label='Max')
ax.scatter(min_kWhs.index, min_kWhs['smoothed'], color='blue', label='Min')
ax.xaxis.set_major_locator(hours)
ax.xaxis.set_major_formatter(h_fmt)
plt.xticks(rotation=45)
plt.ylim(0, 1600)
plt.xlabel('Time')
plt.ylabel('kWh')
plt.title('Daily Load Curve: Smoothed')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(daily_load_1999['mean'], label='Mean', color='black')
ax.fill_between(daily_load_1999.index, daily_load_1999['mean'] + daily_load_1999['std'], daily_load_1999['mean'] - daily_load_1999['std'], color='gray', alpha=0.5, label='Std Dev')
ax.scatter(daily_load_1999.index, daily_load_1999['max'], color='red', label='Max')
ax.scatter(daily_load_1999.index, daily_load_1999['min'], color='blue', label='Min')
ax.xaxis.set_major_locator(hours)
ax.xaxis.set_major_formatter(h_fmt)
plt.xticks(rotation=45)
plt.ylim(0, 1600)
plt.xlabel('Time')
plt.ylabel('kWh')
plt.title('Daily Load Curve: Original Data')
plt.legend()
plt.show()


#%%
#sarimax smoothing
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Define the model
model = SARIMAX(sim_data['kWh'], order=(0, 0, 1), seasonal_order=(1, 0, 1, 24))

# Fit the model
results = model.fit()

# Get the smoothed data
sim_data['smoothed'] = results.fittedvalues
#shift smoothed data up 1 row and add 990 as last value
sim_data['smoothed'] = sim_data['smoothed'].shift(-1)
sim_data['smoothed'][-1] = 990

# Plot the original data and the smoothed data
plt.figure(figsize=(12, 6))
plt.plot(sim_data['kWh'], label='Original Data')
plt.plot(sim_data['smoothed'], label='Smoothed Data')
plt.xlabel('Date')
plt.ylabel('kWh')
plt.title('Simulated Data with SARIMAX Smoothing')
plt.legend()
plt.show()


# plot 2 days
plt.figure(figsize=(12, 6))
plt.plot(sim_data['kWh']['1999-01-01':'1999-01-02'], label='Original Data')
plt.plot(sim_data['smoothed']['1999-01-01':'1999-01-02'], label='Smoothed Data')
plt.xlabel('Date')
plt.ylabel('kWh')
plt.title('Simulated Data with SARIMAX Forecast')
plt.legend()
# %%
print(sim_data)
# %%
from scipy.ndimage import gaussian_filter1d

# Apply Gaussian filter
sim_data['smoothed'] = gaussian_filter1d(sim_data['kWh'], sigma=.75)

# Plot the original data and the smoothed data
plt.figure(figsize=(12, 6))
plt.plot(sim_data['kWh'], label='Original Data')
plt.plot(sim_data['smoothed'], label='Smoothed Data')
plt.xlabel('Date')
plt.ylabel('kWh')
plt.title('Simulated Data with Gaussian Smoothing')
plt.legend()
plt.show()

# Plot 2 days
plt.figure(figsize=(12, 6))
plt.plot(sim_data['kWh']['1999-01-01':'1999-01-02'], label='Original Data')
plt.plot(sim_data['smoothed']['1999-01-01':'1999-01-02'], label='Smoothed Data')
plt.xlabel('Date')
plt.ylabel('kWh')
plt.title('Two Days of Simulated Data with Gaussian Smoothing')
plt.legend()
plt.show()

# %%
print(sim_data)
# %%
