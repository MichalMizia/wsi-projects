from ucimlrepo import fetch_ucirepo


def main():
    # fetch dataset
    wine_quality = fetch_ucirepo(id=186)

    # data (as pandas dataframes)
    X = wine_quality.data.features
    y = wine_quality.data.targets

    print(wine_quality.variables)

    print(X.head())
    print(y.head())


if __name__ == "__main__":
    main()
