#########################################################
#### Convert data in daily sum of Quantity ########
#########################################################

import pandas as pd

# Load the CSV file
file_path = 'C:/Users/Admin/Desktop/MBA FINAL SIP/datas.csv'

# Specify data types for columns if possible
data = pd.read_csv(file_path, low_memory=False)

# Print column names to check for the correct date column name
print(data.columns)

# Ensure your date column is in datetime format, replace 'date_column_name' with actual column name
data['date'] = pd.to_datetime(data['Date'])

# Group by date and sum the quantity, replace 'quantity_column_name' with actual column name
daily_sum = data.groupby(data['date'].dt.date)['Quantity'].sum().reset_index()

# Rename columns for clarity
daily_sum.columns = ['date', 'Quantity']

# Save the result to a new CSV file
output_file_path = 'C:/Users/Admin/Desktop/MBA DTASWR/datas_daily_sum.csv'
daily_sum.to_csv(output_file_path, index=False)

print(f"Daily sum saved to {output_file_path}")


#########################################################
##### Preprocessing and Adjustments ########
#########################################################

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt



# Load your dataset into a DataFrame
dff = pd.read_csv('C:/Users/Admin/Desktop/MBA FINAL SIP/datas_daily_sum.csv')

#########################################################
#### Check Weather The Data Is Stationary Or Not ########
#########################################################

# Create a sample time series dataset
data = {'Date': pd.date_range(start='2022-01-01', periods=100, freq='D'),
        'Quantity': np.random.randn(100)}
dff = pd.DataFrame(data)
dff.set_index('Date', inplace=True)

# Step 1: Visualize the Data
plt.figure(figsize=(12, 6))
plt.plot(dff['Quantity'])
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.show()

##########################################################################

dff = pd.read_csv('C:/Users/Admin/Desktop/MBA FINAL SIP/datas_daily_sum.csv')

dff['date'] = pd.to_datetime(dff['date'])
dff.set_index('date', inplace=True)

# Splitting the dataset into training and testing sets
train_size = int(len(dff) * 0.8)
train, test = dff[:train_size], dff[train_size:]

# Function to calculate RMSE
def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))


window = 7  # for SMA
p, d, q = 1, 1, 1  # for ARIMA

#############################################################################
###################### 1. Simple Moving Average (SMA) #######################
#############################################################################

# 1. Simple Moving Average (SMA)
def simple_moving_average(train, test, window):
    predictions = []
    history = list(train['Quantity'])
    for i in range(len(test)):
        predictions.append(np.mean(history[-window:]))
        history.append(test.iloc[i]['Quantity'])
    return predictions

sma_predictions = simple_moving_average(train, test, window)

print("RMSE for Simple Moving Average:", rmse(test['Quantity'], sma_predictions))

#############################################################################
###############################  2. ARIMA ###################################
#############################################################################

# 2. ARIMA
from statsmodels.tsa.arima.model import ARIMA as new_ARIMA

def arima(train, test, p, d, q):
    # Ensure 'Quantity' column is of numeric data type
    train['Quantity'] = pd.to_numeric(train['Quantity'])
    model = new_ARIMA(train['Quantity'], order=(p, d, q))
    model_fit = model.fit()    
    test['Quantity'] = pd.to_numeric(test['Quantity'])    
    predictions = model_fit.forecast(steps=len(test))
    return predictions

arima_predictions = arima(train, test, p, d, q)

print("RMSE for ARIMA:", rmse(test['Quantity'], arima_predictions))


#############################################################################
###################   3. k-Nearest Neighbors (KNN) ##########################
#############################################################################

# 3. k-Nearest Neighbors (KNN)
def knn(train, test, k=5):
    X_train, y_train = np.arange(len(train)).reshape(-1, 1), train['Quantity'].values
    X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

knn_predictions = knn(train, test)

print("RMSE for k-Nearest Neighbors:", rmse(test['Quantity'], knn_predictions))


#############################################################################
################## 4. Holts Linear Exponential Smoothing ###################
#############################################################################

# 4. Holts Linear Exponential Smoothing

def holt_linear_exponential_smoothing(train, test):
    model = ExponentialSmoothing(train['Quantity'], trend='add', seasonal=None)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test))
    return predictions

holt_linear_predictions = holt_linear_exponential_smoothing(train, test)

print("RMSE for Holt's Linear Exponential Smoothing:", rmse(test['Quantity'], holt_linear_predictions))

#######################################################################
###################        RMSE VALUE     ###########################
#######################################################################

print("RMSE for Simple Moving Average:", rmse(test['Quantity'], sma_predictions))
print("RMSE for ARIMA:", rmse(test['Quantity'], arima_predictions))
print("RMSE for k-Nearest Neighbors:", rmse(test['Quantity'], knn_predictions))
print("RMSE for Holt's Linear Exponential Smoothing:", rmse(test['Quantity'], holt_linear_predictions))


#######################################################################
###################        Saved File      ############################
#######################################################################

import pickle

# Save the Simple Moving Average model using pickle
with open('sma_models.pkl', 'wb') as f:
    pickle.dump(sma_predictions, f)



# Load the Simple Moving Average model from pickle
with open('sma_models.pkl', 'rb') as f:
    sma_models = pickle.load(f)

# Use the loaded model for predictions
predictions_from_pickle = sma_models  # Adjust as needed for your application



#######################################################################################
################################# FORECASTING ###########################################
#########################################################################################

import numpy as np
import pandas as pd
import pickle

# Function to calculate Simple Moving Average
def simple_moving_average(data, window):
    history = data[-window:]
    return np.mean(history)

# Load the Simple Moving Average model from pickle
with open('sma_models.pkl', 'rb') as f:
    sma_models = pickle.load(f)

# Load your dataset into a DataFrame
df = pd.read_csv('C:/Users/Admin/Desktop/MBA FINAL SIP/datas_daily_sum.csv')

# Assuming 'Quantity' column exists in your dataset
# Example: df = pd.DataFrame({'Date': pd.date_range(start='2022-01-01', periods=300, freq='D'), 'Quantity': np.random.randn(300)})

# Sort the dataset by the "date" column (if it's not already sorted)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

# Extract the "Quantity" values
data = df['Quantity'].values

# Forecast the next 30 days using Simple Moving Average
forecast = []
window = 7  # Adjust this based on your model's requirements

for i in range(30):
    next_day_forecast = simple_moving_average(data, window)
    forecast.append(next_day_forecast)
    data = np.append(data, next_day_forecast)

# Create a date range for the next 30 days
last_date = df['date'].max()
forecasted_dates = pd.date_range(start=last_date + pd.DateOffset(1), periods=30, freq='D')

# Create a DataFrame for the forecasted values
forecasted_df = pd.DataFrame({'Date': forecasted_dates, 'Quantity': forecast})

# Convert Quantity to integer (if needed)
forecasted_df['Quantity'] = forecasted_df['Quantity'].astype(int)

# Save the forecasted data to a new CSV file
forecast_csv_filename = 'SMAforecastnewfile.csv'
forecasted_df.to_csv(forecast_csv_filename, index=False)

print(f"Forecasted data saved as {forecast_csv_filename}")




##########################################################################
################# Code To Check SMA model Accuracy########################
##########################################################################


from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

# Function to calculate Simple Moving Average
def simple_moving_average(data, window):
    history = data[-window:]
    return np.mean(history)

# Load your dataset into a DataFrame
df = pd.read_csv('C:/Users/Admin/Desktop/MBA FINAL SIP/datas_daily_sum.csv')

# Assuming 'Quantity' column exists in your dataset
# Example: df = pd.DataFrame({'Date': pd.date_range(start='2022-01-01', periods=300, freq='D'), 'Quantity': np.random.randn(300)})

# Sort the dataset by the "date" column (if it's not already sorted)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date')

# Extract the "Quantity" values
data = df['Quantity'].values

# Forecast the next 30 days using Simple Moving Average
forecast = []
window = 7  # Adjust this based on your model's requirements

for i in range(30):
    next_day_forecast = simple_moving_average(data, window)
    forecast.append(next_day_forecast)
    data = np.append(data, next_day_forecast)

# Create a date range for the next 30 days
last_date = df['date'].max()
forecasted_dates = pd.date_range(start=last_date + pd.DateOffset(1), periods=30, freq='D')

# Create a DataFrame for the forecasted values
forecasted_df = pd.DataFrame({'Date': forecasted_dates, 'Quantity': forecast})

# Calculate accuracy based on a threshold
threshold = 100  # Adjust the threshold as needed

# Classify predictions as 'increase' or 'decrease'
forecasted_df['Prediction_Class'] = np.where(forecasted_df['Quantity'] > threshold, 1, 0)

# Example of actual class labels (you need to define this based on your business logic)
# Assuming you have actual data for the next 30 days to compare with forecasted values
# Replace this with your actual test data when available
actual_data = pd.read_csv('C:/Users/Admin/Desktop/MBA FINAL SIP/SMAforecastnewfile.csv')

# Create actual class labels based on the threshold
actual_data['Actual_Class'] = np.where(actual_data['Quantity'] > threshold, 1, 0)

# Calculate accuracy using accuracy_score
accuracy = accuracy_score(actual_data['Actual_Class'], forecasted_df['Prediction_Class'])

print("Accuracy:", accuracy)

