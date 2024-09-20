import numpy as np
import argparse
import scipy.io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class MyKMeansClustering:
    def __init__(self, dataset_file):
        self.model = None
        self.dataset_file = dataset_file
        self.X = None

        self.read_mat()

    def read_mat(self):
        mat = scipy.io.loadmat(self.dataset_file)
        self.X = mat['X']

    
    def model_fit(self, n_clusters=3, max_iter=300):
        '''
        Initialize self.model here and execute kmeans clustering here
        '''
        self.model = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
        self.model.fit(self.X)

        cluster_centers = self.model.cluster_centers_
        return cluster_centers


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='K-means clustering')
    parser.add_argument('-d','--dataset_file', type=str, default="dataset_q2.mat", help='path to dataset file')
    args = parser.parse_args()
    classifier = MyKMeansClustering(args.dataset_file)
    
    clusters_centers = classifier.model_fit(n_clusters=3)
    print("Cluster Centers:\n", clusters_centers)
    
