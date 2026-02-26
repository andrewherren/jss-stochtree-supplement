# Import Python packages
import pandas as pd
import pyreadr

# Motorcycle dataset
url = "https://github.com/cran/MASS/blob/master/data/mcycle.rda?raw=true"
dst_path = "mcycle.rda"
dst_path_again = pyreadr.download_file(url, dst_path)
res = pyreadr.read_r(dst_path)
mcycle_df = res['mcycle']
mcycle_df.to_csv("data/mcycle.csv", index=False)

# RDD academic probation dataset
url_string = (
    "https://raw.githubusercontent.com/rdpackages-replication/"
    "CIT_2024_CUP/refs/heads/main/CIT_2024_CUP_discrete.csv"
)
data = pd.read_csv(url_string)
data.to_csv("data/CIT_2024_CUP_discrete.csv", index=False)

# ACIC dataset
url_string = "https://raw.githubusercontent.com/andrewherren/acic2024/" \
             "refs/heads/main/data/acic2018/synthetic_data.csv"
df = pd.read_csv(url_string)
df.to_csv("data/acic2018_synthetic_data.csv", index=False)
