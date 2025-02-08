import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):

        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = np.empty([k,1])
        self.labels = []
        self.inputted_mat = np.zeros(shape=(3, 2))

        if type(k) != int:
              raise Exception("k must be an integer")


        if k < 0:
              raise Exception("k has to be a positive integer")

    
    def fit(self, mat: np.ndarray):

        """Given k and epsilon, then:
        1. Initialize m to k random values.
        2. For each data point x, find the closest m_i.
        3. Compute new m_i’s to be the centroid (average) of the closest points
        found in (2).
        4. Compute max change in an m_i from the previous m_i.
        5. Repeat (2) through (4) until the change in centroid is less than some
        epsilon.
        """

        self.inputted_mat = mat
        #find num of observations
        obs_count, feature_count = mat.shape

        #assign random centroids
        curr_centroids = mat[np.random.choice(obs_count, self.k, replace=False)]
        self.centroids = curr_centroids

        for i in range(self.max_iter):
            labels = self.assign_clusters(mat)
            new_centroids = self.find_centroids(mat)
            #find difference in new and old centroids
            centroid_change = np.linalg.norm(new_centroids - curr_centroids)
            #if tolerance level reached, break loop
            if centroid_change < self.tol:
                break
            curr_centroids = new_centroids
        
        self.centroids = curr_centroids
        self.labels = labels


    def predict(self, mat: np.ndarray) -> np.ndarray:
        predicted_clusters =  self.assign_clusters(mat)
        return predicted_clusters
    
    def get_error(self) -> float:

        squared_errors = 0
        for i in range(self.k):
            # get the points in cluster i
            mask = self.labels == i
            cluster_points = self.inputted_mat[mask]
            centroid = self.centroids[i]
            # calculate the squared distance between point and centroid
            squared_distances = np.sum((cluster_points - centroid) ** 2, axis=1)
            squared_errors += np.sum(squared_distances)


        return squared_errors
    
    def get_centroids(self) -> np.ndarray:
        return self.centroids

    def assign_clusters(self, mat: np.ndarray):
        #find distance between centroids and points
        distances = np.linalg.norm(mat[:, np.newaxis] - self.centroids, axis = 2)
        #return lowest distance clusters
        return np.argmin(distances, axis = 1)
    
    def find_centroids(self, mat: np.ndarray):
        new_centroids = np.zeros((self.k, mat.shape[1]))
        #for each cluster
        for i in range(self.k):
            clustered_points = mat[self.labels == i]
            #find new centroid that matches label
            new_centroids[i] = clustered_points.mean(axis=0) if clustered_points.size else self.centroids[i]
        return new_centroids