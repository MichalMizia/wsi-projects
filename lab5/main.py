from ucimlrepo import fetch_ucirepo
from NeuralNetwork import NeuralNetwork
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def main():
    # fetch dataset
    wine_quality = fetch_ucirepo(id=186)

    # data (as pandas dataframes)
    X = wine_quality.data.features
    y = wine_quality.data.targets

    # print(wine_quality.variables)

    # print(X.head())
    # print(y.head())

    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    print(X_train.shape)

    nn = NeuralNetwork(
        input_size=X_train.shape[1], hidden_size=10, output_size=1, lr=0.01
    )
    nn.train(X_train, y_train, 10000)

    predictions = nn.forward_prop(X_test)
    print(predictions)
    mse = mean_squared_error(y_test, predictions)
    # print(f"Mean Squared Error: {mse}")


if __name__ == "__main__":
    main()
