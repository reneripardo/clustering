import matplotlib.pyplot as plt

def plot_kmeans(ans_test_kmeans, values_test, ans_train_kmeans):
    """Plot results.
        Arguments:
            ans_test_kmeans {numpy.array} -- labels of test features.
            values_test {numpy.array} -- features for test.
            ans_train_kmeans {KMeans} -- overall result of kmeans.
        Returns:
            None
    """
    centers = ans_train_kmeans.cluster_centers_
    plt.scatter(values_test[:, 0], values_test[:, 1], c=ans_test_kmeans, s=50)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, label = 'Centroids', alpha=0.5)
    plt.xlabel('SepalLength')
    plt.ylabel('SepalWidth')
    plt.legend()
    plt.show()
    return None