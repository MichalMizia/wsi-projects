import pandas as pd
import numpy as np
import json
import bnlearn as bn


def parse_cpd(cpd):
    new_cpd = {}
    for var, cpd_data in cpd.items():
        # serialize to a dict
        cpd_dict = cpd_data.to_dict()
        new_cpd[var] = cpd_dict

    return new_cpd


def main(config):
    df = pd.read_csv(config["data_file"])

    df = df.head(df.shape[0] // 3)

    # bins = list(range(0, 101, 10))  # discretize age into bins of 10
    # labels = [f"{i}-{i+9}" for i in range(0, 100, 10)]
    # df["Perpetrator Age"] = pd.cut(
    #     df["Perpetrator Age"], bins=bins, labels=labels, right=False
    # )
    # df["Victim Age"] = pd.cut(df["Victim Age"], bins=bins, labels=labels, right=False)

    categorical_columns = [
        "Victim Sex",
        "Victim Race",
        "Victim Age",
        "Perpetrator Sex",
        "Perpetrator Race",
        "Perpetrator Age",
        "Relationship",
        "Weapon",
    ]
    df[categorical_columns] = df[categorical_columns].astype("category")

    model = bn.structure_learning.fit(df, methodtype="hc")

    model = bn.independence_test(model, df, test="chi_square", prune=True)

    model = bn.parameter_learning.fit(model, df)

    G = bn.plot(model, interactive=True)

    if model is None:
        print("Did not manage to train model")
        return

    cpd = bn.print_CPD(model)
    cpd = parse_cpd(cpd)

    with open("output.json", "w") as f:
        json.dump(cpd, f)

    bn.save(model, config["model_file"], overwrite=True)


if __name__ == "__main__":
    config = json.load(open("config.json"))
    main(config)
