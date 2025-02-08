import pytest
import numpy as np
from cluster import KMeans
from cluster import utils
from cluster import Silhouette

def test_kmeans():
    s = Silhouette() 
    kmeans = KMeans(k=5)
    t_clusters, t_labels = utils.make_clusters(scale=2) 
    kmeans.fit(t_clusters)
    klabels = kmeans.labels
    predicted = kmeans.predict(t_clusters)
    scores =  s.score(t_clusters, t_labels)
    utils.plot_multipanel(t_clusters, klabels, predicted, scores)
    pass


def test_2():
    s = Silhouette() 

    kmeans = KMeans(k=3)
    t_clusters, t_labels = utils.make_clusters(scale=0.3) 
    kmeans.fit(t_clusters)
    klabels = kmeans.labels
    predicted = kmeans.predict(t_clusters)
    scores =  s.score(t_clusters, t_labels)
    utils.plot_multipanel(t_clusters, klabels, predicted, scores)
    pass

def test_3():
    s = Silhouette() 

    kmeans = KMeans(k=10)
    t_clusters, t_labels = utils.make_clusters(scale=10) 
    kmeans.fit(t_clusters)
    klabels = kmeans.labels
    predicted = kmeans.predict(t_clusters)
    scores =  s.score(t_clusters, t_labels)
    utils.plot_multipanel(t_clusters, klabels, predicted, scores)
    pass

def test_4():
    try:
        kmeans = KMeans(k=0.3)
        t_clusters, t_labels = utils.make_clusters(scale=10) 
        kmeans.fit(t_clusters)
        klabels = kmeans.labels
        utils.plot_clusters(t_clusters, klabels, "test_3.png")
    except:
        print("k input is invalid")
    pass

def test_5():
    try:
        kmeans = KMeans(k=-9)
        t_clusters, t_labels = utils.make_clusters(scale=10) 
        kmeans.fit(t_clusters)
        klabels = kmeans.labels
        utils.plot_clusters(t_clusters, klabels, "3_test.png")
    except:
        print("k input is invalid")
    pass

"""
test_kmeans()
test_2()
test_3()
test_4()
test_5()
"""
