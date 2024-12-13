from ucimlrepo import fetch_ucirepo  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn import preprocessing  # type: ignore
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score  # type: ignore
from LogisticRegression import LogisticRegression
from typing import List


def main():
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

    # data (as pandas dataframes)
    X = breast_cancer_wisconsin_diagnostic.data.features
    y = breast_cancer_wisconsin_diagnostic.data.targets

    # X.drop(
    #     [
    #         "radius1",
    #         "texture1",
    #         "perimeter1",
    #         "area1",
    #         "smoothness1",
    #         "compactness1",
    #         "concavity1",
    #         "concave_points1",
    #         "symmetry1",
    #         "fractal_dimension1",
    #     ],
    #     axis=1,
    # )

    # map the y data to numbers
    y = y["Diagnosis"].map({"B": 0, "M": 1}).values
    # scale the x data to avoid runtime overflow of sigmoid
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=420
    )

    model = LogisticRegression(learn_rate=0.05, iters=100)
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test, predicted))
    print("F1: ", f1_score(y_test, predicted))
    print("AUROC: ", roc_auc_score(y_test, predicted))


if __name__ == "__main__":
    main()
