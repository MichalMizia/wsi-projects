import json
import bnlearn as bn
import numpy as np
import matplotlib.pyplot as plt


class Generator:
    def __init__(self, model, variables) -> None:
        self.variables = variables
        self.model = model

    def generate_sample(self, df, evidence):
        choice = df.iloc[np.random.choice(df.index, p=df["p"])]
        choice = choice.drop("p")

        evidence.update(choice.to_dict())

        return evidence

    def generate_samples(self, evidence={}, n_samples=1):
        missing_variables = list(set(self.variables) - set(evidence.keys()))
        df = (
            bn.inference.fit(
                self.model,
                variables=missing_variables,
                evidence=evidence,
                verbose=1,
            )
        ).df  # type: ignore

        samples = []
        for _ in range(n_samples):
            evidence_copy = evidence.copy()
            samples.append(self.generate_sample(df, evidence=evidence_copy))
        return samples


if __name__ == "__main__":
    config = json.load(open("config.json"))

    model = bn.load(config["model_file"])

    generator = Generator(
        model,
        variables=[
            "Victim Sex",
            "Victim Race",
            "Victim Age",
            "Perpetrator Sex",
            "Perpetrator Race",
            "Perpetrator Age",
            "Relationship",
            "Weapon",
        ],
    )

    evidence = {
        "Victim Sex": "Male",
        # "Victim Race": "Black",
        # "Victim Age": 20,
        # "Perpetrator Race": "Black",
        # "Perpetrator Sex": "Female",
        "Perpetrator Age": 20,
        "Weapon": "Shotgun",
        # "Relationship": "Husband",
    }

    samples = generator.generate_samples(
        evidence=evidence,
        n_samples=5,
    )

    for sample in samples:
        print("Generated sample:")
        for key, value in sample.items():
            print(f"\t{key}: {value}")

    headers = list(samples[0].keys())
    rows = [list(sample.values()) for sample in samples]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("tight")
    ax.axis("off")
    ax.title.set_text(f"Generated samples with evidence: \n{evidence}")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    n = len(evidence.keys())
    for i in range(n):
        # evidence marked in green
        for j in range(len(rows) + 1):
            cell = table[(j, i)]
            cell.set_facecolor("green")
            cell.set_alpha(0.5)

    plt.show()
