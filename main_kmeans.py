from clustering.ct_processing.ct_cluster import kmeans_cluster
from clustering.ct_processing.ct_io import plot_kmeans

import argparse
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--path_data_set', default='E:\\google_drive\\data_sets\\iris\\iris.data',
                    help='path data set')

    args = vars(ap.parse_args())

    data_set = pd.read_csv(args['path_data_set'])

    min_max_scaler = preprocessing.MinMaxScaler() # init 0 and 1
    array_data_set_scale = min_max_scaler.fit_transform(data_set.values[:,:4])# 0 to 1
    data_set.loc[:,:4] = array_data_set_scale

    data_set.loc[data_set['class'] == 'Iris-setosa', ['class']] = 2
    data_set.loc[data_set['class'] == 'Iris-versicolor', ['class']] = 0
    data_set.loc[data_set['class'] == 'Iris-virginica', ['class']] = 1

    #train and test
    values_train, values_test, class_train, class_test = train_test_split(data_set.values[:,:4], \
                                                                data_set["class"], test_size=0.3)

    ans_test_kmeans, ans_train_kmeans = kmeans_cluster([values_train,class_train], [values_test,class_test])

    list_pred = ans_test_kmeans.tolist()
    list_true = list(class_test.values.copy())
    print("test accuracy: ", accuracy_score(list_true, list_pred))

    plot_kmeans(ans_test_kmeans, values_test, ans_train_kmeans)

if __name__ == "__main__":
    main()