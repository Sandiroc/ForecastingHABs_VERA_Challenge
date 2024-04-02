from datetime import datetime
import pandas as pd
import read_data

# get the data
datestring = datetime.now().strftime("%Y-%m-%d")
file_name_root = "VERA_data" + datestring
read_data.read()

# separate into target vars
TARGET_VAR_NAMES = list("Temp_C_mean")
