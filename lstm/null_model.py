import numpy as np
import sys
import keras
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model

"""
class HABForecastModel():
    # Deep learning architecture with LSTM layer

    
    def __init__(self, forecast_dur: int):
        # Start a sequential model that will allow us to customize layers
        self.model = Sequential()
        self.forecast_horizon = forecast_dur
"""