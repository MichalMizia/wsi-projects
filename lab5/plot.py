from ucimlrepo import fetch_ucirepo
from NeuralNetwork import NeuralNetwork
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Fetch dataset
    wine_quality = fetch_ucirepo(id=186)

    # Data (as pandas dataframes)
    X = wine_quality.data.features
    y = wine_quality.data.targets

    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=1
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    configurations = [
        [10, 6, 4],
        [30, 15, 7],
        [20, 10],
        [10, 6],
        [20],
    ]

    results = []

    for config in configurations:
        mse_list = []
        i = 0
        bad_runs = 0
        while i < 10:
            nn = NeuralNetwork(
                input_size=X_train.shape[1],
                hidden_layers_size=config,
                output_size=1,
                lr=0.1,
                weights_init="HE",
            )
            nn.train(X_train, y_train, 1000)
            predictions = nn.forward_prop(X_test)
            mse = mean_squared_error(y_test, predictions)
            if mse > 3:
                bad_runs += 1
            else:
                i += 1
                mse_list.append(mse)

        mean_mse = np.mean(mse_list)
        best_mse = np.min(mse_list)
        results.append(
            (
                config,
                mean_mse,
                best_mse,
                0 if bad_runs == 0 else bad_runs / (bad_runs + i + 1),
            )
        )

    fig, ax = plt.subplots()
    ax.axis("tight")
    ax.axis("off")
    table_data = [["Configuration", "Mean MSE", "Best MSE", "Bad Runs"]]
    for config, mean_mse, best_mse, bad_runs in results:
        table_data.append(
            [str(config), f"{mean_mse:.4f}", f"{best_mse:.4f}", f"{bad_runs:.4f}"]
        )

    table = ax.table(
        cellText=table_data, loc="center", cellLoc="center", colLabels=None
    )

    plt.title("Neural Network Hidden Layer Configurations and MSE")
    plt.show()


def compare_scalers():
    # Fetch dataset
    wine_quality = fetch_ucirepo(id=186)

    # Data (as pandas dataframes)
    X = wine_quality.data.features
    y = wine_quality.data.targets

    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    scaler = MinMaxScaler()
    X_train_MINMAX = scaler.fit_transform(X_train)
    X_test_MINMAX = scaler.transform(X_test)

    results = []
    mse_list = []
    for _ in range(3):
        nn = NeuralNetwork(
            input_size=X_train_MINMAX.shape[1],
            hidden_layers_size=[30, 15, 7],
            output_size=1,
            lr=0.01,
        )
        nn.train(X_train_MINMAX, y_train, 1000)
        predictions = nn.forward_prop(X_test_MINMAX)
        mse = mean_squared_error(y_test, predictions)
        mse_list.append(mse)

    mean_mse = np.mean(mse_list)
    best_mse = np.min(mse_list)
    results.append(
        (
            "MinMax Scaler",
            mean_mse,
            best_mse,
        )
    )

    # standard
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    mse_list = []
    for _ in range(3):
        nn = NeuralNetwork(
            input_size=X_train.shape[1],
            hidden_layers_size=[30, 15, 7],
            output_size=1,
            lr=0.001,
        )
        nn.train(X_train, y_train, 1000)
        predictions = nn.forward_prop(X_test)
        mse = mean_squared_error(y_test, predictions)
        mse_list.append(mse)

    mean_mse = np.mean(mse_list)
    best_mse = np.min(mse_list)
    results.append(
        (
            "Standar Scaler",
            mean_mse,
            best_mse,
        )
    )

    fig, ax = plt.subplots()
    ax.axis("tight")
    ax.axis("off")
    table_data = [["Configuration", "Mean MSE", "Best MSE"]]
    for config, mean_mse, best_mse in results:
        table_data.append([str(config), f"{mean_mse:.4f}", f"{best_mse:.4f}"])

    table = ax.table(
        cellText=table_data, loc="center", cellLoc="center", colLabels=None
    )

    plt.title("Neural Network Hidden Layer Configurations and MSE")
    plt.show()


if __name__ == "__main__":
    # main()
    compare_scalers()
