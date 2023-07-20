import unittest
from pytools.models import DirichletGPClassifier
from sklearn.datasets import make_classification, make_blobs
import numpy as np
import pandas as pd
import pymc
from bayesian_models.core import distribution
from pytools.utilities import matrix_meshgrid, LinearSpace
import xarray
from arviz import InferenceData


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
        cls.n_classes = len(μ)
        cls.true_means = μ
        cls.true_std = 1
    
    def test_null_api(self):
        '''
            Test blank calls to classifier
        '''
        obj = DirichletGPClassifier(
           
        )
        self.assertTrue(True)
        
    def test_void_priors(self):
        r'''
            Test error raised when failing to specify priors
        '''
        obj = DirichletGPClassifier(
            approximate=False
        )
        self.assertRaises(ValueError, obj.__call__, 
                          TestDirichletGPClassifier.X,
                          TestDirichletGPClassifier.Y
                          )
        
    def test_full_latent(self):
        r'''
            Test model inference without the Hilbert Space approximation
        '''
        X = TestDirichletGPClassifier.X
        y = TestDirichletGPClassifier.Y
        K = TestDirichletGPClassifier.n_classes
        N, M = X.shape
        
        obj = DirichletGPClassifier(
            approximate=False,
            lengthscales = [
                distribution(
                    pymc.Normal, f'ℓ_{i}', 7, 1.5
                    ) for i in range(K)
            ]
        )
        obj(
            X,
            y
        )
        obj.fit(100, tune=100, chains=2)
        
    def test_docstring_demo(self):
        r'''
            Test the docstring demo
        '''
        from sklearn.datasets import make_blobs
        from bayesian_models.core import distribution
        import pandas
        import numpy as np
        
        # Make synthetic data
        x, y = make_blobs()
        cols:list[str] = [
            'Feature_{i}' for i in range(x.shape[1])
        ]
        X, Y = pd.DataFrame(x, columns = cols), pd.DataFrame(y)
        K = len(np.unique(y))
        # Initialize the estimator
        estimator = DirichletGPClassifier(
            hsgp_m = [7]*X.shape[1],
            hsgp_c = 1.3,
            lengthscales = [
                    distribution(
                        pymc.Normal, f'ℓ_{i}', 7, 1.5
                        ) for i in range(K)
                ]
            )
        # Build the graph
        estimator(X,Y)
        # Invoke MCMC for inference
        estimator.fit(
            100, sampler=pymc.sample, tune=100
            )
        spanx = LinearSpace(0,1)
        spany = LinearSpace(0,1)
        Xstar = matrix_meshgrid(spanx, spany,
                                to_pandas=True,
                                columns = cols
                                )
        # Predict on new points
        estimator.predict(Xstar, 
            verbosity = 'full_posterior'
            )
        
    def test_verbosity(self):
        r'''
            Test behavior of the verbosity argument to predict
        '''
        from sklearn.datasets import make_blobs
        from bayesian_models.core import distribution
        import pandas
        import numpy as np
        
        # Make synthetic data
        x, y = make_blobs()
        cols = [
            'Feature_{i}' for i in range(x.shape[1])
        ]
        X, Y = pd.DataFrame(x, columns = cols), pd.DataFrame(y)
        _, M = X.shape
        K = len(Y.iloc[:,0].unique())
        # Initialize the estimator
        estimator = DirichletGPClassifier(
            hsgp_m = [7]*X.shape[1],
            hsgp_c = 1.3,
            lengthscales = [
                distribution(pymc.Normal, f'ℓ_{i}', 7, 1.5) for i in range(K)
                ]
                ,
        )
        # Build the graph
        estimator(X,Y)
        # Invoke MCMC for inference
        estimator.fit(
            100, sampler=pymc.sample, tune=100
            )
        spanx = LinearSpace(0,1)
        spany = LinearSpace(0,1)
        Xstar = matrix_meshgrid(spanx, spany,
                                to_pandas=True,
                                columns = cols
                                )
        # Predict on new points
        full = estimator.predict(Xstar, 
            verbosity = 'full_posterior',
            var_names = ['α_star']
            )
        dist = estimator.predict(Xstar,
                                 verbosity = 'predictive_dist',
                                 var_names = ['α_star'])
        labels = estimator.predict(Xstar, 
                                   verbosity="point_predictions",
                                   var_names = ['α_star'])
        with self.subTest(
            msg="verbosity='full_posterior' returned structure type"):
            self.assertTrue(
                isinstance(full, InferenceData)
            )
        with self.subTest(
            msg="verbosity='full_posterior' returned structure shape"):
            self.assertTrue(all([
                full._groups_all == ['posterior_predictive', 'observed_data', 'constant_data'],
                isinstance(
                    full.posterior_predictive.α_star, xarray.DataArray
                    ),
                 len(full.posterior_predictive.α_star.shape) == 4
            ]))
            
        with self.subTest(
            msg="verbosity='predictive_dist' returned structure type"):
            self.assertTrue(
                isinstance(dist, xarray.Dataset)
            )
        with self.subTest(
            msg="verbosity='predictive_dist' returned structure shape"):
            conds = [
                len(dist.dims) == 3,
                'sample' in dist.dims
                ]
            self.assertTrue(all(conds)
            )
        with self.subTest(
            msg="verbosity='labels' returned structure type"):
            self.assertTrue(
                isinstance(labels, pd.DataFrame)
            )
            
            
    @unittest.skip(
        """
        It not clear at present how to test ground truth for a non parametric model. Some ideas have been proposed. See issue XXXX for details 
        """
        )
    def test_ground_truth(self):
        r'''
            Test the ground truth performance of the classifier
        '''
        # numpyro.set_platform("GPU")
        X,y = TestDirichletGPClassifier.X, TestDirichletGPClassifier.Y
        estimator = DirichletGPClassifier(
            lengthscales = [
                distribution(
                    pymc.Normal, 
                    f"l_{i}",
                    7, 1.5) for i in range(X.shape[1])
            ],
            hsgp_m = [7]*X.shape[1],
            hsgp_c = 1.3
        )
        estimator(
            X, y
        )
        idata = estimator.fit(
            chains=2, 
            # sampler = sample_numpyro_nuts
        )
        
        self.assertTrue(True)