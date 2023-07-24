Visualize Multivariate Models
******************************

In this tutorial we'll see how the visualization utilities :code:`ResponseSurfaceVisualizer` and :code:`ContourSurfaceVisualizer`
can be used the analyze and interpret multidimensional output models. A
good use case for this is visualizing probabilistic classification 
models.

We'll use the simple iris dataset from the :code:`scikit-learn` library
and the custom :code:`DirichletGPClassifier` model for this. We start by
setting up the data in the appropriate format:

.. code-block:: Python3

    from sklearn import datasets
    import pandas as pd

    # Import some data and format them
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns = iris.feature_names)
    y = pd.DataFrame(iris.target)

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

Now that the classifier has been trained, we can visualize how its
predicted probabilities vary across the input space:

.. code-block:: Python3

    from pytools.visualizations import ResponseSurfaceVisualizer
    from pytools.visualizations import ContourSurfaceVisualizer

    # Initialize the visualization
    vizer = ResponseSurfaceVisualizer(
            model = classifier, 
            var_name = "α_star",
            smoothing = [2,2],
            grid = ((0,50,50),(0,50,50) ),
            placeholder_vals = X.mean(axis=0),
            feature_labels=  X.columns,
            predictor_labels = X.columns,
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
        ["0", "1"], 
        X.columns[:-1] 
    )
