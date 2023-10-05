from sklearn.svm import LinearSVC
from datasets import load_dataset


def main():
    # Load training data
    trainset = load_dataset("MuGeminorum/AAL_statistics_volumn",
                            data_files='AAL_statistics_volumn_labelled.csv', split='train[:-33%]')

    testset = load_dataset("MuGeminorum/AAL_statistics_volumn",
                           data_files='AAL_statistics_volumn_labelled.csv', split='train[-33%:]')

    unlabelled_data = load_dataset("MuGeminorum/AAL_statistics_volumn",
                                   data_files='AAL_statistics_volumn_unlabelled.csv', split='train')

    # Preprocess training data
    x_train, y_train, x_test, y_test, x_predict = [], [], [], [], []

    for item in trainset:
        item_vals = list(item.values())
        x_train.append(item_vals[1:-2])
        y_train.append(item_vals[-1])

    for item in testset:
        item_vals = list(item.values())
        x_test.append(item_vals[1:-2])
        y_test.append(item_vals[-1])

    for item in unlabelled_data:
        item_vals = list(item.values())
        x_predict.append(item_vals[1:-2])

    # Train
    clf = LinearSVC(loss="hinge", random_state=42).fit(x_train, y_train)

    # Test
    print('\nAcc: ')
    print(clf.score(x_test, y_test))

    # Predict
    print('\nPredict: ')
    print(clf.predict(x_predict))


if __name__ == "__main__":
    main()
