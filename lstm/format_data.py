from datetime import datetime
from datetime import timedelta
import os
import pandas as pd
import read_data
import sys

def format(model_type, reservoir):
    """Format the data obtained from the VERA website, parameter included depending on what features are included in the lstm"""

    if (model_type != "null" and model_type != "temp_date"):
        raise Exception("Invalid model type input!")

    # get the data
    read_data.read()
    
    # get date
    date = datetime.now()
    datestring = date.strftime("%Y-%m-%d")
        
    # filepath, get csv
    algal_data = pd.read_csv("raw_data/VERA_data_" + datestring + ".csv")
    
    if (model_type == "null"):

        # drop empty observations
        algal_data.dropna(subset=['observation'], inplace = True)

        # get only chloropyhll-a
        algal_data = algal_data[algal_data['variable'] == "Chla_ugL_mean"]
        algal_data.rename(columns={"observation" : "Chla_ugl_mean"}, inplace=True)
        algal_data.drop(columns=["variable"], inplace=True)

    elif (model_type == "temp_date"):
        # drop empty observations
        algal_data.dropna(subset=['observation'], inplace = True)
        # remove extraneous temp values
        algal_data = algal_data[algal_data['depth_m'].isin([1.5, 1.6])]

        # get only chloropyhll-a and temperature
        algal_data = algal_data[algal_data['variable'].isin(['Chla_ugL_mean', 'Temp_C_mean'])]

        # pivot df to separate variables into different columns
        algal_data = algal_data.pivot_table(
        index=['project_id', 'site_id', 'datetime', 'duration', 'depth_m'],
        columns='variable',
        values='observation',
        aggfunc='first'  # Use 'first' to take the first occurrence if there are duplicates
        )

        # Reset the index
        algal_data.reset_index(inplace=True)


    # remove year, only keep month and day
    # algal_data["datetime"] = algal_data["datetime"].str[5:10]

    # get index for split between fcre and bvre
    for ind, value in enumerate(algal_data['site_id']):
        if value =='bvre' and algal_data['site_id'].iloc[ind - 1] == 'fcre':
            split_index = ind
            break

    # split into different datasets
    bvre_chla_data = algal_data[split_index:]
    fcre_chla_data = algal_data[:split_index]

    # reindex dataframes
    bvre_chla_data.reset_index(inplace=True, drop=True)
    fcre_chla_data.reset_index(inplace=True, drop=True)
    
    # send to CSVs
    algal_data.to_csv(path_or_buf="./data/formatted_" + datestring + ".csv", index=False)
    bvre_chla_data.to_csv(path_or_buf="./data/bvre_data" + datestring + ".csv", index=False)
    fcre_chla_data.to_csv(path_or_buf="./data/fcre_data" + datestring + ".csv", index=False)

    # remove previous data
    try:
        today = datetime.today()
        yesterday = today - timedelta(days = 1)

        root = "./data/"
        rm_string = yesterday.strftime("%Y-%m-%d")
        
        os.remove(root + "bvre_data_" + rm_string + ".csv")
        os.remove(root + "fcre_data_" + rm_string + ".csv")
        os.remove(root + "formatted_" + rm_string + ".csv")

    except:
        print("Previous data not initialized")

    
    if (reservoir=="fcre"):
        return fcre_chla_data
    
    elif (reservoir=="bvre"):
        return bvre_chla_data
    
    else:
        raise Exception("Invalid reservoir input!")


if __name__ == "__main__":
    format(sys.argv[1], "fcre")
