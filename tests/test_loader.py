from data_loader import load_data


def main() -> None:
    X_train, y_train, X_test, y_test = load_data()
    assert X_train.ndim == 2 and X_train.shape[1] == 784, "Training samples should be flattened 28x28 inputs"
    assert X_test.ndim == 2 and X_test.shape[1] == 784, "Test samples should be flattened 28x28 inputs"
    assert y_train.shape[1] == 10 and y_test.shape[1] == 10, "Targets must be one-hot encoded"
    assert (y_train.sum(axis=1) == 1).all(), "Each training label should have one-hot sum 1"
    assert (y_test.sum(axis=1) == 1).all(), "Each test label should have one-hot sum 1"
    print("Loader self-test passed: shapes", X_train.shape, y_train.shape, X_test.shape, y_test.shape)


if __name__ == "__main__":
    main()
