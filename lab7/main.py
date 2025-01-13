import pandas as pd
import numpy as np
import json


def main(config):
    df = pd.read_csv(config["data_file"])
    print(df.head())


if __name__ == "__main__":
    config = json.load(open("config.json"))
    main(config)
