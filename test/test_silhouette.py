import pytest
from pytest import approx
import numpy as np
from cluster import Silhouette
from cluster import KMeans
from cluster import utils
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

def test_silhouette():
    s = Silhouette()
    X, y = make_blobs(random_state=42)
    kmeans = KMeans(k=2)
    kmeans.fit(X)
    predicted =  kmeans.predict(X)
    scores = s.score(X, predicted)
    assert(np.average(scores) ==  approx(silhouette_score(X, predicted)))


def test_silhouette_2():
    s = Silhouette()
    t_clusters, t_labels = utils.make_clusters(scale=2) 
    kmeans = KMeans(k=5)
    kmeans.fit(t_clusters)
    predicted =  kmeans.predict(t_clusters)
    scores = s.score(t_clusters, predicted)
    assert(np.average(scores) ==  approx(silhouette_score(t_clusters, predicted)))


def test_silhouette_3():
    s = Silhouette()
    t_clusters, t_labels = utils.make_clusters(scale=0.2) 
    kmeans = KMeans(k=5)
    kmeans.fit(t_clusters)
    predicted =  kmeans.predict(t_clusters)
    scores = s.score(t_clusters, predicted)
    assert(np.average(scores) ==  approx(silhouette_score(t_clusters, predicted)))
