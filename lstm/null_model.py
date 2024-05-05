import numpy as np
import pandas as pd
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import format_data

def get_data(reservoir: str):
    """Get formatted data created by format_data script, indicate which reservoir"""
    
    # get data based on which reservoir is being analyzed
    if (reservoir=='fcre' or reservoir=="bvre"):
        data = format_data.format("null", reservoir)
    
    else:
        raise Exception("Invalid reservoir input!")
    
    return data, reservoir


def normalize_and_format(data: pd.DataFrame):
    """Convert datetime of dataset to datetime type. Min-max normalize chla observations"""
    
    # sort to handle out of place dates
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.sort_values('datetime', inplace=True)

    # min-max normalize observations
    scaler = MinMaxScaler()
    data['Chla_ugL_mean'] = scaler.fit_transform(data['Chla_ugL_mean'].values.reshape(-1,1))
    return data, scaler


def create_sequences(data, seq_length, forecast_horizon):
    """Convert the time series data into sequences, 
    x is a sequence of data from i to the sequence length, 
    while y is just the observation at the i + sequence length"""
    X = []
    y = []
    for i in range(len(data) - seq_length - forecast_horizon + 1):
        X.append(data[i:(i + seq_length)])
        y.append(data[(i + seq_length):(i + seq_length + forecast_horizon)])
    return np.array(X), np.array(y)


def define_model(seq_length, forecast_horizon):
    """Define architecture with lstm layers"""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(forecast_horizon))

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


def epoch_eval(X_test, y_test, model: Sequential, epochs=range(5, 50, 5)):
        
    rmse = list()

    # iterate over each noise
    for i in range(len(epochs)):

        # define and train
        model = define_model(sequence_length)
        model.fit(X_train, y_train, epochs=epochs[i], batch_size=8, verbose=1)

        # Make predictions
        y_pred = model.predict(X_test)

        # Inverse transform the predictions
        y_pred_inverse = scaler.inverse_transform(y_pred.reshape(-1,1))
        y_test_inverse = scaler.inverse_transform(y_test.reshape(-1,1))

        # calculate rmse
        rmse.append(np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse)))

    
    return rmse, epochs

def batch_eval(X_test, y_test, model: Sequential, batches=(2, 4, 8, 16, 32)):
        
    rmse = list()

    # iterate over each noise
    for i in range(len(batches)):

        # define and train
        model = define_model(sequence_length)
        model.fit(X_train, y_train, epochs=30, batch_size=batches[i], verbose=1)

        # Make predictions
        y_pred = model.predict(X_test)

        # Inverse transform the predictions
        y_pred_inverse = scaler.inverse_transform(y_pred.reshape(-1,1))
        y_test_inverse = scaler.inverse_transform(y_test.reshape(-1,1))

        # calculate rmse
        rmse.append(np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse)))

    
    return rmse, batches


def seq_eval(X_test, y_test, model: Sequential, sequences=range(1,101)):
        
    rmse = list()

    # iterate over each noise
    for i in range(len(sequences)):

        # define and train
        model = define_model(sequences[i])
        model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=1)

        # Make predictions
        y_pred = model.predict(X_test)

        # Inverse transform the predictions
        y_pred_inverse = scaler.inverse_transform(y_pred.reshape(-1,1))
        y_test_inverse = scaler.inverse_transform(y_test.reshape(-1,1))

        # calculate rmse
        rmse.append(np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse)))

    
    return rmse, sequences



# ------------------------------------------------------------------------------------------------

# get fcre data
data, reservoir = get_data(reservoir="fcre")

# normalize chl-a observations
data, scaler = normalize_and_format(data)

# Choose sequence length
# look 8 days back, forecast 7 days into the future
sequence_length = 8
forecast_horizon = 7

# Create sequences
X, y = create_sequences(data['Chla_ugL_mean'], sequence_length, forecast_horizon)

# Split the data into training and testing sets

train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape the input data for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# define and train
model = define_model(sequence_length, forecast_horizon)
model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=1)

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform the predictions
y_pred_inverse = scaler.inverse_transform(y_pred)
y_test_inverse = scaler.inverse_transform(y_test.reshape(-1,forecast_horizon))

# Print the predicted chlorophyll-a value for tomorrow
print("Predicted chlorophyll-a value for " + str(forecast_horizon) + " days into the future:", y_pred_inverse[-1])

# save y_test and y_hat
np.savetxt("./predicted.txt", y_pred_inverse)
np.savetxt("./actual.txt", y_test_inverse)
np.savetxt("./xtrain.txt", scaler.inverse_transform(X_test.reshape(-1,1)))

# rmse
print("Model RMSE: ", np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse)))

# plot
plt.figure(figsize=(10, 6))
plt.plot(y_test_inverse, label='Actual')
plt.plot(y_pred_inverse, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Mean Chlorophyll-a content (ug/L)')
plt.title('Actual vs Predicted Chlorophyll-a Values at ' + reservoir)
plt.legend()
plt.savefig("./plots/null_preds_" + reservoir + ".png")

"""
# test noise
noise_rmse, noises = noise_val(X_test, y_test, model)

# plot noise
plt.figure(figsize=(10,6))
plt.plot(noises, noise_rmse)
plt.xlabel('Noise Level')
plt.ylabel('RMSE')
plt.title("RMSE vs. Noise Level")
plt.savefig("./plots/null_noise.png") """

"""
# test epochs
epoch_rmse, epochs = epoch_eval(X_test, y_test, model)

# plot epochs
plt.figure(figsize=(10,6))
plt.plot(epochs, epoch_rmse)
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title("RMSE vs. Epochs")
plt.savefig("./plots/null_epochs.png") """

"""
# test batch size
batch_rmse, batches = batch_eval(X_test, y_test, model)

# plot batches
plt.figure(figsize=(10,6))
plt.plot(batches, batch_rmse)
plt.xlabel('Batches')
plt.ylabel('RMSE')
plt.title("RMSE vs. Batches")
plt.savefig("./plots/null_batches.png") """

"""
# test sequences
batch_rmse, sequences = seq_eval(X_test, y_test, model)

# plot sequences
plt.figure(figsize=(10,6))
plt.plot(sequences, batch_rmse)
plt.xlabel('Sequences')
plt.ylabel('RMSE')
plt.title("RMSE vs. Number of Sequences")
plt.savefig("./plots/null_sequences.png")
"""
