from modelscope.msdatasets import MsDataset
from sklearn.svm import LinearSVC
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")


def main():
    # Load training data
    try:
        trainset = load_dataset(
            "MuGeminorum/AAL_statistics_volumn",
            data_files='AAL_statistics_volumn_labelled.csv',
            split='train[:-20%]'
        )

        testset = load_dataset(
            "MuGeminorum/AAL_statistics_volumn",
            data_files='AAL_statistics_volumn_labelled.csv',
            split='train[-20%:]'
        )

        unlabelled_data = load_dataset(
            "MuGeminorum/AAL_statistics_volumn",
            data_files='AAL_statistics_volumn_unlabelled.csv',
            split='train'
        )
    except ConnectionError:
        dataset = MsDataset.load(
            'MuGeminorum/AAL_statistics_volumn',
            subset_name='default'
        )
        labelled_data = list(dataset['train'])
        unlabelled_data = list(dataset['test'])
        data_len = len(labelled_data)
        p80 = int(data_len * 0.8)
        trainset = labelled_data[:p80]
        testset = labelled_data[p80:]

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
    print(f'\nAcc: {clf.score(x_test, y_test) * 100.0}%')

    # Predict
    print(f'\nPredict: {clf.predict(x_predict)}')


if __name__ == "__main__":
    main()
