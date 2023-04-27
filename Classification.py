from sklearn.svm import LinearSVC
from datasets import load_dataset


def main():
    # Load data
    data = load_dataset("george-chou/Pima")
    trainset = data['train']
    testset = data['test']

    # Preprocess data
    x_train, y_train, x_test, y_test = [], [], [], []

    for item in trainset:
        item_vals = list(item.values())
        x_train.append(item_vals[1:-2])
        y_train.append(item_vals[-1])

    for item in testset:
        item_vals = list(item.values())
        x_test.append(item_vals[1:-2])
        y_test.append(item_vals[-1])

    # Train
    clf = LinearSVC(loss="hinge", random_state=42).fit(x_train, y_train)

    # Test
    print(clf.score(x_test, y_test))


if __name__ == "__main__":
    main()
