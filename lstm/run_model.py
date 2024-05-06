import numpy as np
import sys
from datetime import datetime, timedelta
import pandas as pd
import time
import os
import null_model as nm

NUM_ITER = 1

def binary_conversion(array: np.ndarray):
    """ If chlorophyll-a amount is greater than 20, then an algal bloom will occur """
    binaries = np.zeros_like(array)
    binaries[array > 20] = 1
    return binaries


def generate_predictions(reservoir: str, forecast_len):
    """ Generate the bloom binary mean """
    #start = time.time()
    forecasts = list()

    for i in range(NUM_ITER):
        curr_forecast = nm.forecast(reservoir, forecast_len)
        forecasts.append(binary_conversion(curr_forecast))


    stacked = np.stack(forecasts, axis=0)
    means = np.mean(stacked, axis=0)
    end = time.time()
    #print(end - start)

    return means


def generate_dates(forecast_len):
    """ Generate dates of forecast """
    today = datetime.today() + timedelta(1)
    ref_dates = list()

    for i in range(forecast_len):
        ref_dates.append(today.strftime("%Y-%m-%d"))
        today += timedelta(1)

    return ref_dates


def generate_csv(reservoir: str, forecast_len):
    """ Generate CSV with forecast info for submission """
    model_id = ["protist" for _ in range(forecast_len)]
    dates = generate_dates(forecast_len)
    today = [datetime.today().strftime("%Y-%m-%d") for _ in range(forecast_len)]
    site_id = [reservoir for _ in range(forecast_len)]
    variable = ["Bloom_binary_mean" for _ in range(forecast_len)]
    family = ["bernoulli" for _ in range(forecast_len)]
    parameter = ["prob" for _ in range(forecast_len)]
    prediction = list(generate_predictions(reservoir, forecast_len))

    if reservoir == "fcre":
        depth = [1.6 for _ in range(forecast_len)]
    else:
        depth = [1.5 for _ in range(forecast_len)]

    project_id = ["vera4cast" for _ in range(forecast_len)]
    duration = ["P1D" for _ in range(forecast_len)]

    submission = {
        'model_id': model_id,
        'datetime': dates,
        'reference_datetime': today,
        'site_id': site_id,
        'variable': variable,
        'family': family,
        'parameter': parameter,
        'prediction': prediction,
        'depth_m': depth,
        'project_id': project_id,
        'duration': duration
    }

    # remove submissions past the last 10 days
    directory = "./submissions"
    files = os.listdir(directory)
    file_count = len(files)

    if file_count > 10:
        for file in files:
            file_path = os.path.join(directory, file)
            os.remove(file_path)

    # send to csv
    pd.DataFrame(submission).to_csv("./submissions/vera_submission_" + reservoir + "_" + datetime.today().strftime("%Y-%m-%d") + ".csv")





if __name__ == "__main__":
    generate_csv(sys.argv[1], 7)
