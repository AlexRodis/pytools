import unittest
from pytools.visualizations import ResponseSurfaceVisualizer
from pytools.visualizations import ContourSurfaceVisualizer
from pytools.models import DirichletGPClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from bayesian_models.core import distribution
import numpy as np
import pandas as pd
import pymc


class TestResponseSurfaceVisualizer(unittest.TestCase):
    
    def setUp(self):
        from sklearn.datasets import make_blobs
        mus = np.asarray(
            [[5.0,10.0],[25.0,30.0], [40.0,10.0]])
        sigma = 2.0
        X,Y = make_blobs(
            n_samples=np.asarray([300]*3), n_features=2, centers=mus, cluster_std=4.0
        )
        X = pd.DataFrame(X, columns=[
            f"Feature_{i}" for i in range(X.shape[1])])
        Y = pd.DataFrame(Y)
        return X,Y
    
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
        X,y = self.setUp()
        obj = DirichletGPClassifier(
            approximate=True,
            hsgp_c = 1.2,
            hsgp_m = [7]*2,
            lengthscales=[
                distribution(
                    pymc.Normal, f'l_{i}', 7,1.5,
                    ) for i in range(3)
             
            ]
        )(X,y)
        obj.fit(chains=2)
        obj.save("viz_model.pickle")
        vizer = ResponseSurfaceVisualizer(
            model = obj,
        )

