import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from feature_engine.outliers import Winsorizer

pd.set_option("display.max_columns", None)

data = pd.read_csv(r'D:/Project(Wind turbine)/Project/Wind_turbine.csv', encoding = 'latin-1')
data

data.head(5)
data.tail(5)

data.isna().sum()

data.dtypes

data.duplicated().sum()

sns.boxplot(data)

#making dataframe
df = pd.DataFrame(data)
df

#generating plot
sns.boxplot(data)

#checking outliers
sns.boxplot(data.Wind_speed)
sns.boxplot(data.Power)
sns.boxplot(data.Nacelle_ambient_temperature)
sns.boxplot(data.Generator_bearing_temperature)
sns.boxplot(data.Ambient_temperature)
sns.boxplot(data.Rotor_Speed)
sns.boxplot(data.Nacelle_temperature)
sns.boxplot(data.Bearing_temperature)
sns.boxplot(data.Generator_speed)
sns.boxplot(data.Yaw_angle)
sns.boxplot(data.Wind_direction)
sns.boxplot(data.Wheel_hub_temperature)
sns.boxplot(data.Gear_bix_inlet_temperature)
sns.boxplot(data.Failure_status)

#Detecting outliers and solving them for each and every coloumn
#outliers for Wind_speed
IQR = data['Wind_speed'].quantile(0.75) - data['Wind_speed'].quantile(0.25)

lower_limit = data['Wind_speed'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['Wind_speed'].quantile(0.75) + (IQR * 1.5)
outliers_data_windspeed = np.where(data.Wind_speed > upper_limit, True, np.where(data.Wind_speed < lower_limit, True, False))

#outliers for Power
IQR = df['Power'].quantile(0.75) - df['Power'].quantile(0.25)

lower_limit = df['Power'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Power'].quantile(0.75) + (IQR * 1.5)
outliers_df_power = np.where(df.Power > upper_limit, True, np.where(df.Power < lower_limit, True, False))

#outliers for Nacelle_ambient_temperature
IQR = df['Nacelle_ambient_temperature'].quantile(0.75) - df['Nacelle_ambient_temperature'].quantile(0.25)

lower_limit = df['Nacelle_ambient_temperature'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Nacelle_ambient_temperature'].quantile(0.75) + (IQR * 1.5)
outliers_df = np.where(df.Nacelle_ambient_temperature > upper_limit, True, np.where(df.Nacelle_ambient_temperature < lower_limit, True, False))

#outliers for Generator_bearing_temperature
IQR = df['Generator_bearing_temperature'].quantile(0.75) - df['Generator_bearing_temperature'].quantile(0.25)

lower_limit = df['Generator_bearing_temperature'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Generator_bearing_temperature'].quantile(0.75) + (IQR * 1.5)
outliers_df = np.where(df.Generator_bearing_temperature > upper_limit, True, np.where(df.Generator_bearing_temperature < lower_limit, True, False))

#outliers for Wind_speedGear_oil_temperature
IQR = df['Wind_speedGear_oil_temperature'].quantile(0.75) - df['Wind_speedGear_oil_temperature'].quantile(0.25)

lower_limit = df['Wind_speedGear_oil_temperature'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Wind_speedGear_oil_temperature'].quantile(0.75) + (IQR * 1.5)
outliers_df = np.where(df.Wind_speedGear_oil_temperature > upper_limit, True, np.where(df.Wind_speedGear_oil_temperature < lower_limit, True, False))

#outliers for Nacelle_temperature
IQR = df['Nacelle_temperature'].quantile(0.75) - df['Nacelle_temperature'].quantile(0.25)

lower_limit = df['Nacelle_temperature'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Nacelle_temperature'].quantile(0.75) + (IQR * 1.5)
outliers_df = np.where(df.Nacelle_temperature > upper_limit, True, np.where(df.Nacelle_temperature < lower_limit, True, False))

#outliers for Bearing_temperature
IQR = df['Bearing_temperature'].quantile(0.75) - df['Bearing_temperature'].quantile(0.25)

lower_limit = df['Bearing_temperature'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Bearing_temperature'].quantile(0.75) + (IQR * 1.5)
outliers_df = np.where(df.Bearing_temperature > upper_limit, True, np.where(df.Bearing_temperature < lower_limit, True, False))

#outliers for Generator_speed
IQR = df['Generator_speed'].quantile(0.75) - df['Generator_speed'].quantile(0.25)

lower_limit = df['Generator_speed'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Generator_speed'].quantile(0.75) + (IQR * 1.5)
outliers_df = np.where(df.Generator_speed > upper_limit, True, np.where(df.Generator_speed < lower_limit, True, False))

#outliers for Yaw_angle
IQR = df['Yaw_angle'].quantile(0.75) - df['Yaw_angle'].quantile(0.25)

lower_limit = df['Yaw_angle'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Yaw_angle'].quantile(0.75) + (IQR * 1.5)
outliers_df = np.where(df.Yaw_angle > upper_limit, True, np.where(df.Yaw_angle < lower_limit, True, False))

#outliers for Wind_direction
IQR = df['Wind_direction'].quantile(0.75) - df['Wind_direction'].quantile(0.25)

lower_limit = df['Wind_direction'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Wind_direction'].quantile(0.75) + (IQR * 1.5)
outliers_df = np.where(df.Wind_direction > upper_limit, True, np.where(df.Wind_direction < lower_limit, True, False))

#outliers for Wheel_hub_temperature
IQR = df['Wheel_hub_temperature'].quantile(0.75) - df['Wheel_hub_temperature'].quantile(0.25)

lower_limit = df['Wheel_hub_temperature'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Wheel_hub_temperature'].quantile(0.75) + (IQR * 1.5)
outliers_df = np.where(df.Wheel_hub_temperature > upper_limit, True, np.where(df.Wheel_hub_temperature < lower_limit, True, False))

#outliers for Gear_bix_inlet_temperature
IQR = df['Gear_bix_inlet_temperature'].quantile(0.75) - df['Gear_bix_inlet_temperature'].quantile(0.25)

lower_limit = df['Gear_bix_inlet_temperature'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Gear_bix_inlet_temperature'].quantile(0.75) + (IQR * 1.5)
outliers_df = np.where(df.Gear_bix_inlet_temperature > upper_limit, True, np.where(df.Gear_bix_inlet_temperature < lower_limit, True, False))

#outliers for Failure_status
IQR = df['Failure_status'].quantile(0.75) - df['Failure_status'].quantile(0.25)

lower_limit = df['Failure_status'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Failure_status'].quantile(0.75) + (IQR * 1.5)
outliers_df = np.where(df.Failure_status > upper_limit, True, np.where(df.Failure_status < lower_limit, True, False))


# outliers data for Wind_speed
data_out = data.loc[outliers_data_windspeed, ]
#Trimming/removing the outliers from the data
data_trimmed_windspeed = data.loc[~(outliers_data_windspeed), ]
data_trimmed_windspeed.shape, data_trimmed_windspeed.shape

# outliers data for Wind_speed
df_out = df.loc[outliers_df_windspeed, ]
#Trimming/removing the outliers from the data
df_trimmed_windspeed = df.loc[~(outliers_df_windspeed), ]
df.shape, df_trimmed_windspeed.shape
