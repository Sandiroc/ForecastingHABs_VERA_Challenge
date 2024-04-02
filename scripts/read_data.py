from datetime import datetime
import pandas as pd

def read():
    # localize data directly from VERA site
    url = "https://renc.osn.xsede.org/bio230121-bucket01/vera4cast/targets/project_id=vera4cast/duration=P1D/daily-insitu-targets.csv.gz"
    data = pd.read_csv(url)

    date = datetime.now()
    datestring = date.strftime("%Y-%m-%d")
    data.to_csv("./data/VERA_data_" + datestring + ".csv", index=False)



if __name__ == "__main__":
    read()