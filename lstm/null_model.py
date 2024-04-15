import numpy as np
import sys
import pandas as pd
import keras
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import format_data

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
    """Convert the time series data into sequences"""
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def define_model(seq_length):
    """Define architecture with lstm layers"""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ------------------------------------------------------------------------------------------------

# get fcre dat
data = get_data(reservoir="fcre")

# normalize chl-a observations
data, scaler = normalize_and_format(data)

# Choose sequence length
sequence_length = 10

# Create sequences
X, y = create_sequences(data['Chla_ugl_mean'], sequence_length)

# Split the data into training and testing sets
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape the input data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# define and train
model = define_model(sequence_length)
model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform the predictions
y_pred_inverse = scaler.inverse_transform(y_pred.reshape(-1,1))
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1,1))

# Print the predicted chlorophyll-a value for tomorrow
print("Predicted chlorophyll-a value for tomorrow:", y_pred_inverse[-1])

# plot
plt.figure(figsize=(10, 6))
plt.plot(y_test_inverse, label='Actual')
plt.plot(y_pred_inverse, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Chlorophyll-a Value')
plt.title('Actual vs Predicted Chlorophyll-a Values')
plt.legend()
plt.show()

np.savetxt("./something.csv", y_pred_inverse)
np.savetxt("./something1.csv", y_test_inverse)