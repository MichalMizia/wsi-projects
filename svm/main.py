import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from SVM import SVM


def generate_linearly_separable_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = np.array([1 if x[0] + x[1] > 0 else -1 for x in X])
    return X, y


def generate_concentric_data(n_samples=200, noise=0.5):
    np.random.seed(0)
    n_samples_per_class = n_samples // 2

    r_inner = 1
    theta_inner = (
        np.linspace(0, 2 * np.pi, n_samples_per_class)
        + np.random.randn(n_samples_per_class) * 0.1
    )
    X_inner = np.column_stack(
        [r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)]
    )
    X_inner += noise * np.random.randn(n_samples_per_class, 2)
    y_inner = np.ones(n_samples_per_class)

    r_outer = 3
    theta_outer = np.linspace(0, 2 * np.pi, n_samples_per_class)
    X_outer = np.c_[
        r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)
    ]  # c_ = column_stack
    X_outer += noise * np.random.randn(n_samples_per_class, 2)
    y_outer = -np.ones(n_samples_per_class)

    # Combine the data
    X = np.vstack((X_inner, X_outer))
    y = np.concatenate((y_inner, y_outer))

    return X, y


def plot_data(X, y, svm):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color="red", label="Class -1")
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color="blue", label="Class 1")

    # Create a grid to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = svm.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.contourf(
        xx,
        yy,
        Z,
        alpha=0.3,
        levels=np.linspace(Z.min(), Z.max(), 3),
        cmap="Paired",
    )
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="k")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Linearly Separable Data with Decision Boundary")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    X, y = generate_concentric_data()

    # svm = SVM()
    svm = SVM(kernel="rbf", gamma=1)
    svm.fit(X, y, iters=1000)

    # test the accuracy:
    y_pred = svm.predict(X)
    y_pred = np.where(y_pred > 0, 1, -1)
    accuracy = np.mean(y == y_pred)
    print(f"Accuracy: {accuracy}")

    plot_data(X, y, svm)
