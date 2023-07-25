import unittest
from pytools.visualizations import ResponseSurfaceVisualizer
from pytools.visualizations import ContourSurfaceVisualizer
from pytools.models import DirichletGPClassifier
from sklearn.datasets import make_blobs, load_iris
from sklearn.model_selection import train_test_split
from bayesian_models.core import distribution
import numpy as np
import pandas as pd
import pymc


# NOTE: This could be optimized by training models during setup once,
# pickling them and ensuring deletion afterwards. Anything other than
# ground truth testing does not require MCMC reruns

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
        cls.Y = pd.DataFrame(Y, columns="target")
        
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
            predictor_labels = ["target"],
            feature_labels = ["Feature_0", "Feature_1"],
        )
        
    def test_no_errors(self):
        
        iris = load_iris()
        X = pd.DataFrame(
            data = iris.data,
            columns = iris.feature_names
        )
        y = pd.DataFrame(
            data = iris.target
        )
        n_classes:int = len(y.iloc[:,0].unique()) # 3 classes 
        n_features:int = X.shape[-1]
        classifier = DirichletGPClassifier(
        hsgp_c = 1.3,
        hsgp_m = [7]*n_features, # Approximation settings
        lengthscales = [
            distribution(
                pymc.Normal, f"l_{i}", 7,1.5
                ) for i in range(n_classes)
            ]
        )(X, y)
        classifier.fit(chains=2)
        vizer = ResponseSurfaceVisualizer(
            model = classifier, 
            var_name = "α_star",
            smoothing = [2,2],
            grid = ((0,50,50),(0,50,50) ),
            placeholder_vals = X.mean(axis=0),
            feature_labels=  X.columns,
            predictor_labels = [0,1,2],
            colormaps = [
                'viridis', 'magma', 'tealrose', 
                "inferno", "blues"
                ],
            scaling_factor = .8,
            colorbar_spacing_factor= .05,
            colorbar_location = .8,
            layout = dict(),
            adaptable_zaxis = False,
            autoshow=False
        )
        # Call the visualizer with output coordinate names to plot
        fig=vizer(
            [0, 1], 
            X.columns[:-2] 
        )
        self.assertTrue(True)
        
    def test_raises_no_feature_coords_call(self):
                
        iris = load_iris()
        X = pd.DataFrame(
            data = iris.data,
            columns = iris.feature_names
        )
        y = pd.DataFrame(
            data = iris.target
        )
        n_classes:int = len(y.iloc[:,0].unique()) # 3 classes 
        n_features:int = X.shape[-1]
        classifier = DirichletGPClassifier(
        hsgp_c = 1.3,
        hsgp_m = [7]*n_features, # Approximation settings
        lengthscales = [
            distribution(
                pymc.Normal, f"l_{i}", 7,1.5
                ) for i in range(n_classes)
            ]
        )(X, y)
        classifier.fit(chains=2)
        vizer = ResponseSurfaceVisualizer(
            model = classifier, 
            var_name = "α_star",
            smoothing = [2,2],
            grid = ((0,50,50),(0,50,50) ),
            placeholder_vals = X.mean(axis=0),
            feature_labels=  X.columns,
            colormaps = [
                'viridis', 'magma', 'tealrose', 
                "inferno", "blues"
                ],
            scaling_factor = .8,
            colorbar_spacing_factor= .05,
            colorbar_location = .8,
            layout = dict(),
            adaptable_zaxis = False,
            autoshow=False
        )
        # Call the visualizer with output coordinate names to plot
        with self.assertRaises(ValueError):
            fig=vizer(
                ["0", "1"], 
                X.columns[:-2] 
            )


    def test_raises_no_predictor_coords(self):
                
        iris = load_iris()
        X = pd.DataFrame(
            data = iris.data,
            columns = iris.feature_names
        )
        y = pd.DataFrame(
            data = iris.target
        )
        n_classes:int = len(y.iloc[:,0].unique()) # 3 classes 
        n_features:int = X.shape[-1]
        classifier = DirichletGPClassifier(
        hsgp_c = 1.3,
        hsgp_m = [7]*n_features, # Approximation settings
        lengthscales = [
            distribution(
                pymc.Normal, f"l_{i}", 7,1.5
                ) for i in range(n_classes)
            ]
        )(X, y)
        classifier.fit(chains=2)
        vizer = ResponseSurfaceVisualizer(
            model = classifier, 
            var_name = "α_star",
            smoothing = [2,2],
            grid = ((0,50,50),(0,50,50) ),
            placeholder_vals = X.mean(axis=0),
            
            feature_labels=  X.columns,
            predictor_labels = ["0","1","2"],
            colormaps = [
                'viridis', 'magma', 'tealrose', 
                "inferno", "blues"
                ],
            scaling_factor = .8,
            colorbar_spacing_factor= .05,
            colorbar_location = .8,
            layout = dict(),
            adaptable_zaxis = False,
            autoshow=False
        )
        # Call the visualizer with output coordinate names to plot
        with self.assertRaises(ValueError):
            fig=vizer(
                ["0", "1"], 
                X.columns[:-2] 
            )