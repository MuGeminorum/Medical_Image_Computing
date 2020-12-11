import os
from csv import reader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import LinearSVC

def main():
    # Load data
    with open("data/pima.csv") as f:
        csv_data = reader(f, delimiter=',')
        raw_data = np.array(list(csv_data))

    # Preprocess data
    data_x = []
    data_y = []
    tuple_len = len(raw_data[0])
    for i in raw_data:
        if not i:
            continue
        data_x.append([float(j) for j in i[0:tuple_len - 2]])
        if i[tuple_len - 1] == "yes":
            data_y.append(1)
        else:
            data_y.append(0)

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=73)

    #TODO:
    clf = LinearSVC(loss="hinge", random_state=42).fit(x_train, y_train)
    print(clf.score(x_test, y_test))

if __name__ == "__main__":
    main()