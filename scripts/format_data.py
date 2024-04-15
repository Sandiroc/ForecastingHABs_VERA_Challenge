from datetime import datetime
import pandas as pd
import read_data

def format(model_type, reservoir):
    """Format the data obtained from the VERA website, parameter included depending on what features are included in the lstm"""

    if (model_type != "null"):
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

        # remove year, only keep month and day
        # algal_data["datetime"] = algal_data["datetime"].str[5:10]

        # get index for split between fcre and bvre
        for ind, value in enumerate(algal_data['site_id']):
            if value =='bvre' and algal_data['site_id'].iloc[ind - 1] == 'fcre':
                split_index = ind
                break

        bvre_chla_data = algal_data[split_index:]
        fcre_chla_data = algal_data[:split_index]
        
        # send to CSVs
        algal_data.to_csv(path_or_buf="./data/formatted_" + datestring + ".csv", index=False)
        bvre_chla_data.to_csv(path_or_buf="./data/bvre_data" + datestring + ".csv", index=False)
        fcre_chla_data.to_csv(path_or_buf="./data/fcre_data" + datestring + ".csv", index=False)

        if (reservoir=="fcre"):
            return fcre_chla_data
        
        elif (reservoir=="bvre"):
            return bvre_chla_data
        
        else:
            raise Exception("Invalid reservoir input!")


if __name__ == "__main__":
    format("null", "fcre")
