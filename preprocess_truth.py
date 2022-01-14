#!/usr/bin/env python3
import pandas as pd

# Load the data
url_new_cases = "https://raw.githubusercontent.com/covid19-forecast-hub-europe/covid19-forecast-hub-europe/main/data-truth/JHU/truth_JHU-Incident%20Cases.csv"
url_new_deaths = "https://raw.githubusercontent.com/covid19-forecast-hub-europe/covid19-forecast-hub-europe/main/data-truth/JHU/truth_JHU-Incident%20Deaths.csv"

df_deaths = pd.read_csv(url_new_deaths)
df_cases = pd.read_csv(url_new_cases)

# Merge into a single dataframe
df_all = df_cases.merge(df_deaths, how='inner', on=["location_name", "date"])

# Extract data for Slovenia only
df_slo = df_all[df_all['location_name'] == 'Slovenia'].reset_index(drop=True)
df_slo['date'] = pd.to_datetime(df_slo['date'])

# Get weekly sums and add week of the year
df_weekly = df_slo.resample('W-Sat', on='date').sum().reset_index().sort_values(by='date')
df_weekly['EpiWeek'] = df_weekly['date'].dt.isocalendar().week

# Output formatting
df_weekly.drop(['date'], axis=1, inplace=True)
df_weekly = df_weekly[['EpiWeek', 'value_x', 'value_y']]
df_weekly.columns = ['EpiWeek', 'NewCases', 'NewDeaths']

# Save to CSV
df_weekly.to_csv("covid.slovenia.week.csv", index=True)
