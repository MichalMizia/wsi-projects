import pandas as pd
import numpy as np
import json


def parse_data():
    df = pd.read_csv("data/US_Crime_DataSet.csv")
    df = df[
        [
            "Victim Sex",
            "Victim Age",
            "Victim Race",
            "Perpetrator Sex",
            "Perpetrator Age",
            "Perpetrator Race",
            "Relationship",
            "Weapon",
        ]
    ]
    df = df[
        ~df.apply(
            lambda row: row.astype(str).str.contains("unknown", case=False).any(),
            axis=1,
        )
    ]
    df = df[(df["Victim Age"] != 0) & (df["Perpetrator Age"] != 0)]
    df.to_csv("data/US_Crime_Data.csv", index=False)


if __name__ == "__main__":
    parse_data()
