import numpy as np
import pandas as pd
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import format_data
import csv
from datetime import datetime, timedelta, date


def get_data(reservoir: str):
    """Get formatted data created by format_data script, indicate which reservoir"""
    
    # get data based on which reservoir is being analyzed
    if (reservoir=='fcre'):
        data = format_data.format("null", reservoir)
    
    elif (reservoir=="bvre"):
        data = format_data.format("null", reservoir)

    else:
        raise Exception("Invalid reservoir input!")

    return data


def normalize_and_format(data: pd.DataFrame):
    """Convert datetime of dataset to datetime type. Min-max normalize chla observations"""
    
    # sort to handle out of place dates
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.sort_values('datetime', inplace=True)

    # min-max normalize observations
    scaler = MinMaxScaler()
    data['Chla_ugl_mean'] = scaler.fit_transform(data['Chla_ugl_mean'].values.reshape(-1,1))
    return data, scaler


def create_sequences(data, seq_length):
    """Convert the time series data into sequences, 
    x is a sequence of data from i to the sequence length, 
    while y is just the observation at the i + sequence length"""
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def define_model(seq_length, forecast_dur):
    """Define architecture with lstm layers"""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(forecast_dur))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def add_gaussian_noise(X, noise):
    """Add noise to a vector"""
    mean = 0
    std_dev = noise * (np.max(X) - np.min(X))
    gaussian_noise = np.random.normal(mean, std_dev, X.shape)

    return X + gaussian_noise


def noise_val(X_test, y_test, model: Sequential, noises=(0.01, 0.05, 0.1, 0.15, 0.2)):
    """Test robustness of model with different levels of noise"""


    rmse = list()

    # iterate over each noise
    for i in range(len(noises)):

        # add noise to test data
        X_test = add_gaussian_noise(X_test, noises[i])
        y_test = add_gaussian_noise(y_test, noises[i])

        # predict
        y_pred = model.predict(X_test)

        # Inverse transform the predictions
        y_pred_inverse = scaler.inverse_transform(y_pred.reshape(-1,1))
        y_test_inverse = scaler.inverse_transform(y_test.reshape(-1,1))

        # calculate rmse
        rmse.append(np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse)))

    
    return rmse, noises


# ------------------------------------------------------------------------------------------------

# get fcre data
data = get_data(reservoir="fcre")

# normalize chl-a observations
data, scaler = normalize_and_format(data)

# Choose sequence length
sequence_length = 35

# Create sequences
X, y = create_sequences(data[['Chla_ugl_mean', 'Temp_C_mean']], sequence_length)

# Split the data into training and testing sets
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape the input data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 2))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 2))

# define and train
model = define_model(sequence_length, forecast_dur=35)
model.fit(X_train, y_train, epochs=15, batch_size=8, verbose=1)

#Creating a list to store all the predictions
all_predictions = []

#Running the model 100 times for uncertainty purposes
for i in range(100):
    # Make predictions
    y_pred = model.predict(X_test)
    # Inverse transform the predictions
    y_pred_inverse = scaler.inverse_transform(y_pred.reshape(-1,1))
    all_predictions.append(y_pred_inverse[-1])

# Convert the list to a numpy array for easier manipulation
all_predictions = np.array(all_predictions)

# Calculating the percentage of predicted values over 20
count = len([i for i in all_predictions if i >= 20])
uncertainty = (count / len(all_predictions)) * 100


# Make predictions
y_pred = model.predict(X_test)

y_test_inverse = scaler.inverse_transform(y_test.reshape(-1,1))

# Print the predicted chlorophyll-a value for tomorrow
print("Predicted chlorophyll-a value for 35 days into the future:", y_pred_inverse[-1])

# Converting the y_pred_inverse into probability using bernoulis distribution
y_pred_prob = 1/(1+np.exp(-y_pred_inverse))

# save y_test and y_hat
np.savetxt("./predicted.txt", y_pred_inverse)
np.savetxt("./probabilities.txt", y_pred_prob)
np.savetxt("./actual.txt", y_test_inverse)
np.savetxt("./xtrain.txt", scaler.inverse_transform(X_test.reshape(-1,1)))

# rmse
print("Model RMSE: ", np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse)))
print("Uncertainty: ", uncertainty)
# plot
plt.figure(figsize=(10, 6))
plt.plot(y_test_inverse, label='Actual')
plt.plot(y_pred_inverse, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Mean Chlorophyll-a content (ug/L)')
plt.title('Actual vs Predicted Chlorophyll-a Values')
plt.legend()
plt.savefig("./plots/preds.png")

# test noise
noise_rmse, noises = noise_val(X_test, y_test, model)

# plot noise
plt.figure(figsize=(10,6))
plt.plot(noises, noise_rmse)
plt.xlabel('Noise Level')
plt.ylabel('RMSE')
plt.title("RMSE vs. Noise Level")
plt.savefig("./plots/rmse.png")

start_date = datetime.now().date()  # Start from today
dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(35)]

#row_list = ["project_id", "model_id", "datetime", "reference_datetime", "duration", "site_id", "family", "parameter", "variable", "prediction"]
df = pd.DataFrame(columns=["project_id", "model_id", "datetime", "reference_datetime", "duration", "site_id", "depth_m", "family", "parameter", "variable", "prediction"], size=(35, 11))

# Assign values to each column row by row
for date in dates:
    df = df.append({
        "project_id": "vera4cast",
        "model_id": "protist",
        "datetime": date,
        "reference_datetime": date.today(),
        "duration": "P1D",
        "site_id": "fcre",
        "depth_m": "1.6",
        "family": "bernoulli",
        "parameter": "prob",
        "variable": "Bloom_binary_mean",
        "prediction": y_pred_prob[index]
    }, ignore_index=True)

# Write to a csv file
df.to_csv("./raw_data/forecast_" + start_date.strftime('%Y-%m-%d') + ".csv", index=False)