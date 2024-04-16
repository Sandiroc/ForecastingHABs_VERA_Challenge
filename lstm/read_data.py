from datetime import datetime
from datetime import timedelta
import os
import pandas as pd

def read():
    """Read data in from the VERA website"""
    
    # localize data directly from VERA site
    url = "https://renc.osn.xsede.org/bio230121-bucket01/vera4cast/targets/project_id=vera4cast/duration=P1D/daily-insitu-targets.csv.gz"
    data = pd.read_csv(url)

    date = datetime.now()
    datestring = date.strftime("%Y-%m-%d")
    data.to_csv("./raw_data/VERA_data_" + datestring + ".csv", index=False)

    try:
        # remove yesterday's raw data
        today = datetime.today()
        yesterday = today - timedelta(days = 1)

        root = "./raw_data/VERA_data_"
        rm_string = yesterday.strftime("%Y-%m-%d")
        
        os.remove(root + rm_string + ".csv")

    except: 
        print("Yesterday data not initialized")




if __name__ == "__main__":
    read()