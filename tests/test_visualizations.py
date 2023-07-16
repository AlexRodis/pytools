import unittest
from pytools.visualizations import ResponseSurfaceVisualizer
from pytools.visualizations import ContourSurfaceVisualizer
from pytools.models import DirichletGPClassifier
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd


class TestResponseSurfaceVisualizer(unittest.TestCase):
    
    @classmethod
    def setupClass(cls):
        from sklearn.datasets import make_blobs
        mus = np.asarray(
            [[5.0,10.0],[25.0,30.0], [40.0,10.0]])
        sigma = 2.0
        X,Y = make_blobs(
            n_samples=np.asarray([300]*3), n_features=2, centers=mus, cluster_std=4.0
        )
        cls.X = pd.DataFrame(X, columns=[
            f"Feature_{i}" for i in range(X.shape[1])])
        cls.Y = pd.DataFrame(Y)
        
    def test_api_blank(self):
        obj = DirichletGPClassifier()

# class TestContourSurfaceVisualizer(unittest.TestCase):
#     pass


# class TestVizFunctions(unittest.TestCase):
#     pass


