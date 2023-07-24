Multiclass Classification with the DGP Model
============================================

The :code:`pytools.models.DirichletGPClassifier` model can be used for
multiclass classification with improved interperability and uncertainty
estimates.

We'll work with the well-known iris dataset

.. code-block:: Python3

    from sklearn import datasets
    import pandas as pd

    # Import some data and format them
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns = iris.feature_names)
    y = pd.DataFrame(iris.target)

This dataset is a quite simple classification problem with 3 classes
(the species)

.. code-block:: Python3

    X.head()
    # >>Output:
    #        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
    # 0                5.1               3.5                1.4               0.2
    # 1                4.9               3.0                1.4               0.2
    # 2                4.7               3.2                1.3               0.2
    # 3                4.6               3.1                1.5               0.2
    # 4                5.0               3.6                1.4               0.2


.. code-block:: Python3

    y.head()
    # >>Output:
    #    0
    # 0  0
    # 1  0
    # 2  0
    # 3  0
    # 4  0

We'll use the provided :code:`DirichletGPClassifier` for this classification
problem:


.. code-block:: Python3

    from pytools.models import DirichletGPClassifier
    import pymc
    from bayesian_models.core import distribution

    n_classes:int = len(y.iloc[:,0].unique()) # 3 classes 

    # Set classification hyperparameters and settings
    classifier = DirichletGPClassifier(
        hsgp_c = 1.3,
        hsgp_m = [7]*X.shape[-1], # Approximation settings
        lengthscales = [
            distribution(
                pymc.Normal, f"l_{i}", 7,1.5
                ) for i in range(n_classes)
            ]
    )
    # Initialize the classifier
    classifier(X,y)
    # Fit / Infer with MCMC
    classifier.fit(chains=2)

Initializing and calling :code:`fit` is all that's needed to fit the 
classifier to the data. We can predict on unseen/new points, via the 
:code:`DirichletGPClassifier.predict` method:

.. code-block:: Python3

    import numpy as np
    Xnew = pd.DataFrame(
        data = np.random.rand(10,X.shape[-1])
        columns = X.columns
        )
    preds = classifier.predict(
        Xnew, 
        verbosity = "point_predictions",
        var_names = ["α_star"]
        )
    print(preds.head())
    # >> Output:
    #   0
    # 0  1
    # 1  2
    # 2  1
    # 3  0
    # 4  1

