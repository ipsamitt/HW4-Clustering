import pytest
import numpy as np
from cluster import Silhouette
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

s = Silhouette()
X, y = make_blobs(random_state=42)
kmeans = KMeans(k=2)
kmeans.fit(X)
predicted =  kmeans.predict(X)
print(silhouette_score(X, predicted))
print(s.score(X, predicted))