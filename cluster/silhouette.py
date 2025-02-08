import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:

        n = X.shape[0]
        scores = np.zeros(n)

        dist_mat = self.pairwise_dist(X)

        for i in range(n):
            curr_cluster = y[i]
            cluster_distances = dist_mat[i][y == curr_cluster]
            a_i = np.mean(cluster_distances[cluster_distances != 0])
            
            different_cluster_distances = []
            for label in np.unique(y):
                if label != curr_cluster:
                     # compute distance to points in the other cluster
                    other_cluster_points = dist_mat[i][y == label]
                    different_cluster_distances.append(np.mean(other_cluster_points))

            b_i = min(different_cluster_distances)  # nearest cluster's average distance
           
            # calculate silhouette score for point i
            if a_i < b_i:
                s_i = (b_i - a_i) / b_i
            elif a_i == b_i:
                s_i = 0
            else:
                s_i = (b_i - a_i) / a_i

            scores[i] = s_i
        
        return scores
   
    #compute distance matrix of pairwise distances of X
    def pairwise_dist(self, X: np.ndarray) -> np.ndarray:

        n_samples = X.shape[0]        
        distance_matrix = np.zeros((n_samples, n_samples))
        
        # compute the pairwise distances
        for i in range(n_samples):
            for j in range(i + 1, n_samples):  
                # compute euclidean distance between  i and j
                dist = np.linalg.norm(X[i] - X[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist  

        return distance_matrix


