from sklearn.svm import LinearSVC
from modelscope.msdatasets import MsDataset
from datasets import load_dataset


def main():
    # Load data
    try:
        trainset = load_dataset("MuGeminorum/Pima", split="train")
        testset = list(load_dataset("MuGeminorum/Pima", split="validation")) + \
            list(load_dataset("MuGeminorum/Pima", split="test"))
    except ConnectionError:
        trainset = MsDataset.load(
            'MuGeminorum/Pima', subset_name='default', split='train')
        testset = list(MsDataset.load('MuGeminorum/Pima', subset_name='default', split='test')) + \
            list(MsDataset.load('MuGeminorum/Pima',
                 subset_name='default', split='validation'))

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
    clf = LinearSVC(loss="hinge", random_state=42,
                    max_iter=700000).fit(x_train, y_train)

    # Test
    print(clf.score(x_test, y_test))


if __name__ == "__main__":
    main()
