import unittest
from pytools.models import DirichletGPClassifier
from sklearn.datasets import make_classification, make_blobs
import numpy as np
import pandas as pd

class TestDirichletGPClassifier(unittest.TestCase):
    
    @classmethod
    def setUp(cls):
        μ = np.asarray([
            [5.0,5.0],[5.0,10.0], [15.0,5.0],[15.0,10.0],
            ])
        σ = 1.0
        X, Y = make_blobs(
            centers = μ, cluster_std = σ 
            )
        N, M = X.shape
        cls.X = pd.DataFrame(X, columns = [
            "feature_{i}" for i in range(M)
        ] )
        cls.Y = pd.DataFrame(Y, columns = ["target"] )
    
    def test_null_api(self):
        '''
            Test blank calls
        '''
        obj = DirichletGPClassifier()