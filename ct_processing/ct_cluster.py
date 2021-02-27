from sklearn.cluster import KMeans

def kmeans_cluster(list_train,list_test):
    """Apply kmeans.
        Arguments:
            list_train {list} -- list of arrays (features and class).
            list_test {list} -- list of arrays (features and class).
        Returns:
            inf_test_kmeans {numpy.array} -- labels of test features.
            inf_train_kmeans {KMeans} -- overall result of kmeans.
    """

    kmeans = KMeans(n_clusters=3, init='random')
    inf_train_kmeans = kmeans.fit(list_train[0])
    inf_test_kmeans = kmeans.predict(list_test[0])

    return inf_test_kmeans, inf_train_kmeans