#   Copyright 2023 Alexander Rodis
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# This module contains additional math functions and definitions, mostly
# related to neural network activations / transfer functions.
# The following functions are implemented:
# - ReLu := Rectified Linear Units, parametric or Leaky
# - ELU
# - | SWISS := The swiss activation function, with fixed non learnable
#     parameter. More accuratly "SWISS-1"
# - GELU 
# - SiLU

import pymc as pm, pymc
import arviz as az
import numpy as np
import typing
import pandas as pd
from dataclasses import dataclass, field
from .utilities import numpy_replace
from pytools.utilities import Pipeline
from bayesian_models.core import Distribution


class BayesianModel(typing.Protocol):
    r'''
        Base Protocol for Bayesian Estimators.

        Methods:
        =========

            - | __call__(X:pd.DataFrame)->Self := Specify the
                full probability model for inference. Create
                the :code:`pymc.Model` object and associated
                graph

            - | fit(*args:tuple, sampler:Callable=pymc.sample, 
                **kwargs:dict[str,Any])->arviz.InferenceData :=
                Perform inference on specified model and data
                using MCMC. Implementation is specified by the
                :code:`sampler` arguement. All other arguments
                are forwarded to the sampler Callable. 
                :code:`infer` is an alias for :code:`fit`

            - | predict(self,Xnew:pandas.DataFrame, 
                verbosity_level:int=0)->arviz.InferenceData :=
                Predict on new points :code:`Xnew` with a trained
                model. :code:`verbosity_level` determines optional
                postprocessing logic. Should at least have two levels
                :code:`verbosity_level=0` should return "point 
                predictions" as appropriate for the model. 
                :code:`verbosity_level=1` should return the full
                posterior predictive trace. Other levels can be defined
                as appropriate the model

            - | save(self, save_dir:Optional[str]=None, 
                method:str="pickle")->Self := Save the model for later
                reuse. :code:`save_dir` should be an object attribute
                set at other call time or during initialization. When
                provided to the method, should override the attribute.
                :code:`method` should be either "pickle" or "netcdf".
                In the former case the third party :code:`cloudpickle`
                package is used to serialize the entire object for 
                rapid reuse (model should have an :code:`idata` field
                for posterior data). When :code:`method="netcdf"`
                should store only the :code:`idata` attribute via
                :code:`arviz.InferenceData().to_netcdf()`

            - | load(save_dir:str=None, 
                method:str="pickle")->Self := Load a pretrained model
                from memory. When :code:`method="pickle"`, unpickles
                a serialized object and returns the new object. When
                :code:`method="netcdf"`, creates a fresh instance with
                the specified posterior trace. No checks are being made
                for posterior trace consistency with the specified model

            - | plot_trace(self, *args, **kwargs) := Wrapper for 
                :code:`arviz.plot_trace`

            - | plot_posterior(self, *args, **kwargs) := Wrapper for
                :code:`arviz.plot_posterior`

            - | plot_energy(self, *args, **kwargs) := Wrapper for
                :code:`arviz.plot_energy`

            - | summary(self, *args, **kwargs) := Wrapper for 
                :code:`arviz.summary`
    '''

    def __call__(self, X:pd.DataFrame):
        raise NotImplemented()

    def fit(self, *args:tuple,sampler:typing.Callable, 
            **kwargs:dict[str,typing.Any])->az.InferenceData:
        raise NotImplemented()

    def predict(self, Xnew:pd.DataFrame, *args:tuple, 
                **kwargs:dict[str,typing.Any])->az.InferenceData:
        raise NotImplemented()

    def save(self, save_dir:typing.Optional[str]=None, 
             method:str="pickle")->None:
        raise NotImplemented()
                 
    def load(self,save_dir:str, method:str="pickle"):
        raise NotImplemented()

    def plot_trace(self,*args:tuple,**kwargs:dict):
        raise NotImplemented()

    def plot_posterior(self, *args, **kwargs):
        raise NotImplemented()

    def plot_energy(self, *args:tuple, **kwargs:dict):
        raise NotImplemented()

    def summary(self, *args:tuple, **kwargs:dict)->pd.DataFrame:
        raise NotImplemented()

@dataclass(slots=True, kw_only=True)
class BayesianEstimator:
    save_dir:typing.Optional[str] = None
    idata:typing.Optional[az.InferenceData] = field(
        repr=False, init =False, default=None
        )
    model:typing.Optional[pymc.Model] = field(
        repr=False, init =False, default=None
        )
    posterior:typing.Optional[az.InferenceData] = field(
        repr=False, init =False, default=None
        )
    coords:dict[str, typing.Any] = field(
        init=False, default_factory=dict
        )


    def __raise_uninitialized__(self):
        r'''
            Dry method for raising on an uninitialized model
        '''
        if not self._initialized:
            raise RuntimeError((
                "Model has not been initialized. Ensure that the object"
                " has been called first"
            ))
            
    def __raise_untrained__(self):
        r'''
            Dry method for raising on an untrained model
        '''
        if not self._trained:
            raise RuntimeError((
                "Model has not been trained. Ensure that the object's "
                "fit method has been called"
            ))
            
    def __raise_any__(self):
        r'''
            DRY method to raise if uninitialized of untrained
        '''
        self.__raise_uninitialized__()
        self.__raise_untrained__()
    
    
    def plot_posterior(self, *args:tuple, **kwargs:dict):
        r'''
            Convenience wrapper for `arviz.plot_posterior`

            Raises:
            =======

                - | RuntimeError := If the :code:`_trained` or
                    :code:`_intialized` sentinels are False
        '''
        self.__raise_untrained__()
        return az.plot_posterior(self.idata, *args, **kwargs)

    
    def plot_trace(self, *args, **kwargs):
        r'''
            Convenience wrapper for `arviz.plot_trace`

            Raises:
            =======

                - | RuntimeError := If the :code:`_trained` or
                    :code:`_intialized` sentinels are False
        '''
        self.__raise_uninitialized__()
        return az.plot_trace(self.idata, *args, **kwargs)

    def summary(self, *args, **kwargs):
        r'''
            Convenience wrapper for `arviz.summary`

            Raises:
            =======

                - | RuntimeError := If the :code:`_trained` or
                    :code:`_intialized` sentinels are False
        '''
        return az.summary(self.idata, *args, **kwargs)

    def plot_energy(self, *args:tuple, **kwargs:dict[str,typing.Any]):
        r'''
            Plot MCMC transition energy

            Wrapper around :code:`arviz.plot_energy`. All arguments are forwarded
        '''

        return az.plot_energy(self.idata)

    def __pickle__(self):
        r'''
            Save the entire model with the :code:`cloudpickle`
            module
        '''
        import cloudpickle
        with open(self.save_dir, "wb") as file:
            file.write(cloudpickle.dumps(self) )
        return self

    def __save_netcdf__(self):
        self.idata.to_netcdf(self.save_dir)
    
    def save(self, save_dir:typing.Optional[str], 
             method:str="pickle"):
        r'''
            Save the entire model for later reuse

            Args:
            =====

                - | save_dir:str := Directory to save the model
                    to. Optional. Must be specified as either an
                    argument to this method or as an attribute of
                    the object. If set via this method updates the
                    object's `save_dir` attribute for late reuse
                    prior to saving

                - | method:str='pickle' := Select how the model should
                    save. Accepted values are 'pickle' and 'netcdf'. For
                    `method='pickle'`, serialize the entire model object
                    as a string. For `method='netcdf'` save the
                    `arviz.InferenceData` object only. Optional.
                    Defaults to 'pickle'

            Raises:
            =======

                - | ValueError := If supplied method argument is not one
                    of :code:'pickle' or :code:'netcdf'
        '''
        if self.save_dir is None and save_dir is None:
            raise ValueError((
                "`save_dir` must be provided as either an "
                "object attribute or an argument to `save`. "
            ))
        if save_dir is not None:
            self.save_dir = save_dir
        else:
            save_dir = self.save_dir
        if method == "pickle":
            self.__pickle__()
        elif method == "netcdf":
            self.__save_netcdf__()
        else:
            raise ValueError((
                "Unrecognized save method. Expected on of 'pickle' or "
                f"'netcdf' but received method = {method} instead"
            ))

    @staticmethod
    def load(save_dir:str, method:str='pickle',
            X_train=None,Y_train=None):
        r'''
        Load a classifier from memory

        Args:
        =====

            - | save_dir:str := Directory to load the model from

            - | method:str='pickle' := How the model was saved. If
                `method:str='pickle'` unpickles the entire object return
                a new instance. If `method:str='netcdf'`, creates a new
                instance with the specified inference data. When
                `method:str='netcdf'` the training data must also be
                supplied

            - | X_train:Optional[pandas.DataFrame=None] := Training
                features. Only required if `method:str='netcdf'`,
                otherwise ignored

            - | Y_train:Optional[pandas.DataFrame=None] := Training
                targets. Only required if `method:str='netcdf'`,
                otherwise ignored
                
    '''
        if method == "pickle":
            import cloudpickle
            with open(save_dir, 'rb') as file:
                return cloudpickle.loads(file.read())
        elif method == "netcdf":
            if X_train is None or Y_train is None:
                raise ValueError((
                    "When loading a model with `method='netcdf'` "
                    "the training data must be supplied"
                ))
            idata = az.from_netcdf(save_dir)
            obj = DirichletGPClassifier(
                pipeline=self.pipeline
            )
            obj.save_dir=save_dir
            obj(X_train, Y_train)
            obj.idata= idata
            obj._intialized=True
            obj._trained=True
            obj.__post_fit__()
            return obj


@dataclass(slots=True, kw_only=True)
class DirichletGPClassifier(BayesianEstimator):
    r'''
        Gaussian Process Classifier using the Dirichlet distribution as
        a likelihood over observations. The model specification is as
        follows:

        .. math::
                \begin{array}{c}
                    \mathbb{D} = \{(\mathbf{x}_i,y_i)|\mathbf{x}_i\in 
                    \mathcal{X}\in\reals^d, y\in\{0,1,\dots,K \}\}\\
                    \\
                    \mathbf{f} \thicksim \begin{pmatrix}
                        \mathcal{GP}_0(\mu_0(\mathbf{x}), 
                        \kappa_0(\mathbf{x}))
                        & \dots & \mathcal{GP}_K(\mu_K(\mathbf{x}), 
                        \kappa_K(\mathbf{x}))
                    \end{pmatrix}\\
                    \\
                    \mathbf{\alpha} = e^{\mathbf{f}}\\
                    \\
                    y \thicksim Dir(\mathbf{\alpha})
                    \\
                \end{array}

        Object Public Attributes:
        ==========================

            - | pipeline:Pipeline := An object with similar
                functionality to :code:`sklearn.Pipeline` but only for
                preprocessing steps. Defines a series of objects with
                :code:`fit_transform` and :code:`transform` methods, for
                data preprocessing (i.e rescaling, dimensionality
                reduction, etc). Optional

            - | features := A :code:`numpy.typing.NDArray` containing
                labels for the input features. Inferred from passed
                data (columns of predictor matrix)

            - | classes := A :code:`numpy.typing.NDArray` containing
                unique classes to predict. Inferred from training data
                as unique values of target matrix

            - | model:Optional[pymc.Model] := A reference to the
                :code:`pymc.Model` object

            - | idata:Optional[arviz.InferenceData] := Posterior samples
                from MCMC inference

            - | posterior:Optional[arviz.InferenceData] := Posterior
                predictive samples. Stores the values from the most
                recent call to :code:`predict` for efficient reuse

            - | approximate:bool=True := If :code:`True` use the Hilbert
                Space approximation for faster inference. Only effective
                for up to 3-4 input dimensions. Optional. Defaults to
                :code:`True`

            - | hsgp_m:Optional[Sequence[int]] := The number of bases
                functions to use for the approximation. Must be a
                sequence of positive integers whose length exactly
                matches the number of features in  the data. The more
                bases functions are used, the more accurate the
                approximation and higher the computational cost
            
            - | hsgp_c:Optional[float]=1.2 := Length factor for the
                Hilbert Space approximation. For HSGP processes the
                approximation is valid in a subspace of the input space
                of the form c*[-L, +L] where L is the maximum absolute
                value observed, dimension wise. Mainly effects the
                behavior of the approximation near the edges of the
                subspace

            - | perturbation_factor:float=1e-6 := A small perturbation
                factor for observed points. Since the likelihood of the
                Dirichlet is :math:`+\infty` at simplex edges, we shift
                the observations but this small factor to improved
                numerical stability. The one hot encoded target
                variables (...,1,0,...) are perturbed to (...,
                1-perturbation_factor, perturbation_factor/K-1), where K
                is the length of the encoding vector

            - | save_dir:Optional[str] := Directory to save the model
                to. Optional. Can be supplied during object
                initialization or during calls to the :code:`save`
                method
                
            - | posterior_probabilities:bool=True := If :code:`True`
                includes the deterministic variable \pi_star in the
                posterior trace, which returns predicted probabilities
                for all the classes. Optional. Defaults to :code:`True`.
                Set to False to reduce memory consumption
            
            - | posterior_distribution:bool=True := If :code:`True`
                includes the deterministic variable \alpha_star in the
                posterior trace, which returns the predictive Dirichlet
                distribution. From this uncertainties, precision and
                predictions can be extracted. Optional and defaults to
                :code:`True`. Set to :code:`False` to reduce memory
                consumption
            
            - | posterior_labels:bool=True := If :code:`True` includes
                the Deterministic variable y_star in the posterior
                trace, which returns predicted class labels (integer
                encoded). Optional and defaults to :code:`True`. Set to
                :code:`False` to reduce memory consumption

        Object Private Attributes:
        ==========================

            - | _encodings:Optional[dict[str,Any]] := A dictionary
                mapping categories integers

            - | _decodings:Optional[dict[str, Any]] := A dictionary
                (reverse) mapping of integer encodings of categories to
                their labels. Note the integers here are recorded as
                strings for compatibility reasons

            - | _n_features:Optional[int] := The number of input
                feautures (prior to any embedding)

            - | _n_inputs:Optional[int] := The number of input features
                post embedding. Used internally to specify tensor
                shapes, active dimensions etc

            - | _n_obs:Optional[int] := The total number of observations

            - | _n_classes:Optional[int] := Number of unique target
                values

            - | _classes:Optional[list[str]] := An list of unique target
                values

            - | _coords:Optional[dict[str,Any]] := A coordinate
                dictionary mapping dimensions to their coordinates.
                Currently unused

            - | _processor:Optional[Any] := A reference to the specific
                Gaussian Process implementation. Is set to either
                :code:`pymc.go.HSGP` when :code:`approximate=True` or
                :code:`pymg.gp.Latent` when :code:`approximate=False`

            - | _initialized:bool=False := Model specification sentinel.
                Prevents calls to post-__call__ methods until the
                :code:`__call__` method has been called

            - | _trained:bool=False := Model inference sentinel.
                Prevents calls to post-fit methods until the :code:`fit`
                method has been called

        Object Public Methods:
        ========================

            - | __init__(pipeline:Pipeline)->self. := An instance of
                `Pipeline` encapsulating data preprocessing operations

            - | __call__(X:pandas.DataFrame,y:pandas.DataFrame)->self :=
                Specify the full probability model for inference,
                providing inputs :code:`X` and outputs :code:`y`

            - | fit(sampler:Callable=pymc.sample, *sampler_args:tuple,
                **sampler_kwargs:dict[str,Any])->arviz.InferenceData :=
                Perform inference on the data, using MCMC. Returns a
                :code:`arviz.InferenceData` object containing the
                results of inference. :code:`sampler` is a Callable
                specifying which implementation of MCMC to use. Defaults
                to :code:`pymc.sample`. Other options include
                :code:`pymc.sample_numpyro_nuts`, which requires
                external dependencies

            - | predict(Xnew:pandas.DataFrame, *args:tuple,
                verbosity_level:int='point_predictions',
                kwargs:dict[str,Any])->arviz.InferenceData := Predict on
                new points :code:`Xnew`, with optional postprocessing
                logic defined by the :code:`verbosity_level` argument.
                For :code:`verbosity_level='full_posterior'` simply
                returns the posterior predictive trace. For
                :code:`verbosity_level='predictive_dist'` stacks chains
                and extracts the 'posterior_predictive' group. For
                :code:`verbosity_level='point_predictions'` returns only
                the predicted labels. The following options must be
                true:
                
            - | save(save_dir:Optional[str]=None,
                method:str='pickle')->None := Save the model into target
                directory. :code:`save_dir` must be provided, either as
                an object attribute or during the call. For
                code:`method='pickle'` will serialize the entire object
                for ready reuse. For :code:`method=netcdf` only saves
                the posterior code:`InferenceData` object. When loading
                the latter from memory no checks are being made that the
                posterior trace is consistent with the model object.

            - | load(save_dir:str,
                method:str='pickle')->DirichletGPClassifier := Load a
                model from memory according to the method specified by
                :code:`method`. For :code:`method='pickle'` simply
                unserializes the object and returns a the new instance
                of :code:`DirichletGPClassifier`

            - | plot_trace(*args:tuple, **kwargs:dict)-> := Display a
                trace plot of model inference. Wrapper for
                :code:`arviz.plot_trace`. All arguments are forwarded.             

            - | plot_energy(*args:tuple,
                **kwargs:dict)->pandas.DataFrame := Display an energy
                plot for model inference. Wrapper for
                :code:`arviz.plot_energy`. All arguments are forwarded.

            - | plot_posterior(*args:tuple, **kwargs:dict)-> := Display
                a plot of the models posterior. Wrapper for
                :code:`arviz.plot_posterior`. All arguments are
                forwarded.            
    '''
    pipeline:Pipeline = field(default= None)
    features:typing.Optional = field(init =False, default=None)
    classes:typing.Optional = field(init =False, default=None)
    model:typing.Optional[pymc.Model] = field(
        repr=False, init = False, default=None
        )
    lengthscales:typing.Optional[
        typing.Sequence[Distribution]
        ] = field(default = None)
    means:typing.Optional = field(default=None)
    idata:typing.Optional[az.InferenceData] = field(
        repr=False, init =False, default=None
        )
    posterior:typing.Optional[az.InferenceData] = field(
        repr=False, init =False, default=None
        )
    approximate:bool=True
    hsgp_c:typing.Optional[float] = field(default=1.3)
    hsgp_m:typing.Optional[typing.Sequence[int]] = field(
        default_factory=lambda : [7]
        )
    perturbation_factor:float= 1e-6
    trace:typing.Optional[az.InferenceData] = field(
        init=False, default=None
        )
    save_dir:typing.Optional[str] = None
    posterior_distribution:bool = True
    posterior_probabilities:bool = True
    posterior_labels:bool = True
    
    _encodings:typing.Optional[dict] = None
    _decodings:typing.Optional[dict] = None
    _n_features:typing.Optional[int] = field(
        init = False, default = None
        )
    _n_inputs:typing.Optional[int] = field(
        init = False, default = None
        )
    _n_obs:typing.Optional[int] = field(
        init = False, default = None
        )
    _n_classes:typing.Optional[int] = field(
        init = False, default = None
        )
    _classes:typing.Optional = field(default = None)
    _means:typing.Optional[np.typing.NDArray] = None
    _target_label:typing.Optional[str] = None
    _coords:typing.Optional[dict] = field(
        repr=False, init=False, default_factory=dict
        )
    _processor:typing.Optional = field(init=False, repr=False)
    _prior_latents:typing.Optional[list] = field(
        repr= False, init=False, default_factory=list
        )
    _conditional_latents:typing.Optional[list] = field(repr= False, init=False, default_factory=list)
    _processes:typing.Optional[list] = field(
        repr= False, init=False, default_factory=list
        )
    _initialized:bool = False
    _trained:bool = False
    _3d_cache:typing.Optional=field(
        init=False, repr=False, default=None
        )
    _default_colors:typing.Optional[list[str]] = field(
        repr=False, init=False, default=None
        )
    _linspace:typing.Optional[typing.NamedTuple] = field(
        repr=False, init=False, default=None
        ) 

    def __post_init__(self)->None:
        r'''
            Initialize the gaussian processor
        '''
        from collections import namedtuple
        from functools import partial
        if self.approximate:
            self._processor = partial(
                pymc.gp.HSGP, c = self.hsgp_c, m = self.hsgp_m
            )
        else:
            self._processor = pymc.gp.Latent

        self._default_colors= [
            'plasma','viridis','blues','aggrnyl', 'agsunset', 'algae', 
            'amp', 'armyrose', 'balance',
            'blackbody', 'bluered', 'blugrn', 'bluyl', 'brbg',
            'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
            'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
            'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
            'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
            'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
            'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
            'piyg',  'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
            'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
            'rdgy', 'twilight','rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
            'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
            'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
            'turbo',  'ylgn', 'ylgnbu', 'ylorbr',
            'ylorrd'
                ]

        self._linspace = namedtuple(
            'LinearSpace', ['start', 'stop', 'n_points']
            )
        

    def __preprocess_targets__(self, Y_raw):
        r'''
            Transform categorical targets to a shape, appropriate
            for inference. The categorical target is first
            on-hot-encoded and the resulting proability vector
            is perturbed by a factor of `perturbation_factor`, as
            follows:
            
            .. math::  
                y^{\prime} = \begin{pmatrix}
                \dots & 0 & 1 & 0& \dots
                \end{pmatrix} \rightarrow
                \overset{\thicksim}{y} = \begin{pmatrix}
                \dots & \frac{\epsilon}{K} & 1-\epsilon & 
                \frac{\epsilon}{K}& \dots
                \end{pmatrix}
        '''
        self._encodings = {v:k for k, v in  enumerate(self._classes)}
        self._decodings = {str(v):k for k,v in self._encodings.items()}
        y_enc = Y_raw.replace(self._encodings)
        y_dir=pd.get_dummies(y_enc).astype(int).replace(
            {0:self.perturbation_factor/self._n_classes , 1: 1-self.perturbation_factor}
        ).values
        return y_dir

    def __preprocess_features__(self, X_raw, transform_only:bool=False):
        r'''
            Prepear features for inference by applying the selected
            preprocessing pipeline

            Args:
            ======

                - | X_raw:pandas.DataFrame := The raw input data

                - | transform_only:bool=False := If `True` only
                    tranform the data, do not fit the pipeline

            Returns:
            ========

                - | X_trans:pandas.DataFrame := The processed input
                    data
        '''
        self._means= X_raw.values.mean(axis=0, keepdims=True)
        if self.pipeline is not None:
            if not transform_only:
                X_trans = self.pipeline.fit_transform(X_raw)
            else:
                X_trans = self.pipeline.transform(X_raw)
        else:
            X_trans = X_raw
        return X_trans

    def __validate_priors__(self):
        if self.means is None:
            warn((
                "Mean function not explicitly specified. A centered "
                "parameterization will be used"
            ))
            self.means = [pymc.gp.mean.Constant(c=0)]*self._n_classes
        if len(self.means) == 1:
            self.means *= self._n_classes
            
        if len(self.means) != self._n_classes:
            raise ValueError((
                "Mean must be a sequence of length 1 on exactly match the "
                f"number of unique classes. Saw {self._n_classes} classes "
                f"but received {len(self.means)} mean instead instead"
            ))
        if self.lengthscales is None:
            raise ValueError((
                "Priors on lengthscales not specified. Expected "
                "a `bayesian_models.Distribution` object but received "
                "None instead"
            ))
        if len(self.lengthscales) == 1:
            self.lengthscales *= self._n_classes
        if len(self.lengthscales) != self._n_classes:
            raise ValueError((
                "Lengthscales must be a sequence whose length "
                "exactly matches the number of classes in the "
                f"data. Detected {self._n_classes} classes but "
                f"received {len(self.lengthscales)} priors instead"
            ))
            

    def __preprocess_data__(self, X_raw, 
                            Y_raw)->tuple[
            np.typing.NDArray, np.typing.NDArray]:
        r'''
            Preprocess features and targets prior
            to model inference

            Args:
            =====

                - | X_raw:pandas.DataFrame := The raw input data

                - | Y_raw:pandas.DataFrame := The raw targets

            Returns:
            ========

                - | processed_data:tuple[pandas.DataFrame,
                    pandas.DataFrame] := A tuple of the process
                    input features and targets
        '''
        if isinstance(X_raw, pd.DataFrame):
            self._coords = dict(
                index = X_raw.index,
                features = X_raw.columns,
            )
        else:
            self._coords = dict(
                index = np.arange(0,X_raw.shape[0]),
                columns = np.arange(0, X_raw.shape[1]),
            )
        # If the user passes a slice that exluded certain categories
        # the pandas index will keep track of the abscent categories
        # and one hot encoding will treat them as present. We reset
        # the index to correctly infer the number of distinct classes
        Y_raw = Y_raw.astype(str).astype('category')
        self._n_obs, self._n_features = X_raw.shape
        classes = sorted(Y_raw.iloc[:,0].unique().to_list())
        self._n_classes = len(classes)
        self._classes = classes
        self._target_label = Y_raw.columns[0]
        self.__validate_priors__()
        X_trans = self.__preprocess_features__(X_raw)     
        Y_trans = self.__preprocess_targets__(Y_raw)
        return X_trans, Y_trans
        

    def __call__(self, X_train, Y_train):
        r'''
            Initialize the classifier by specifying the
            full probability model

            Args:
            =====

                - | X_train:pandas.DataFrame := The model input feauters
                    to be used for inference

                - | Y_train:pandas.DataFrame := The model target to be
                    used for inference. Expected to be a 2D
                    single-column matrix with the categorical variable
                    of interest. All encodings are handled internally

            Returns:
            =========

                - self := The object itself for method chaining
        '''
        X_pr, Y_pr = self.__preprocess_data__(
            X_train, Y_train
        )
        with pymc.Model(coords=self._coords) as model:
            train_inputs = pymc.ConstantData('train_inputs', X_pr )
            train_outputs = pymc.ConstantData('train_outputs', Y_pr)
            N,M = train_inputs.shape.eval()
            self._n_inputs=M
            for id, lengthscale, mean in zip(
                self._classes, self.lengthscales, 
                self.means, strict=True
                ):
                ℓ = lengthscale.dist(
                    lengthscale.name, *lengthscale.dist_args, 
                    **lengthscale.dist_kwargs, 
                    shape = (self._n_inputs,)
                )
                κ_se = pymc.gp.cov.ExpQuad(self._n_inputs, ls=ℓ )
                κ = κ_se
                μ = mean
                gp = self._processor(
                    mean_func=μ, 
                    cov_func=κ
                )
                f = gp.prior(f'_f_{id}', train_inputs)
                self._processes.append(gp)
                self._prior_latents.append(f)
            f = pytensor.tensor.stack(self._prior_latents).T
            α = pymc.Deterministic('α', pymc.math.exp(f) )
            y = pymc.Dirichlet('y', observed = train_outputs,a = α)
        self.model = model
        self._initialized = True
        return self

    def __post_fit__(self)->None:
        r'''
            Post inference actions

            Prepear the model for predictions by building the
            conditional distribution. Constructs additional nodes in the
            computation graph for predictions on new points. Creates a
            data node for unseen points and duplicates all GP nodes to
            generate the corresponding conditional distributions.

            Conditional variables are always named with suffix "_star",
            for example "f_star", "α_star" and "y_star".
            
            The following variables are added to the model:
            
                - f_star := The 'raw' conditional distribution of the GP
                
                - | α_star := The exponentiated output of the GP,
                    defining the predicted Dirichlet distributions.
                
                - | π_star := The discrete predicted probabilities
                    predicted. Only added if
                    :code:`posterior_proabilities` is :code:`True`.
                    Optional and defaults to :code:`True`. Disable for
                    reduced memory consumption of the posterior
                    predictives' trace.
                    
                - | y_star := The predicted class label, encoded as an
                    integer. Only added if :code:`posterior_labels` is
                    :code:`True`. Optional and defaults to :code:`True`.
                    Disable to reduce the posterior predictives' memory
                    footprint
        '''
        with self.model:
            inputs = pymc.MutableData('inputs', np.random.rand(3,self._n_inputs))
            for id,gp in zip(self._classes,self._processes):
                f_star=gp.conditional(f'f_star_{id}',inputs)
                self._conditional_latents.append(f_star)
            f_star = pytensor.tensor.stack(self._conditional_latents).T
            if self.posterior_distribution:
                α_star = pymc.Deterministic(
                    'α_star', pymc.math.exp(f_star) )
            if self.posterior_probabilities:
                π_star = pymc.Dirichlet('π_star', a=α_star)
            if self.posterior_labels:
                y_star = pymc.Categorical('y_star', p=π_star)
            

    # NOTE: This method is nearly identical across all models, since
    # it relies of pymc. Should be refactored into a parent class,
    # taking into account the new __post_fit__ method
    def fit(self, *args, sampler:typing.Callable=pymc.sample,
            **kwargs)->az.InferenceData:
        r'''
            Perform inference on the model with the given data. Only
            inference with Hamiltonian Monte Carlo is currently
            supported.

            Args:
            =====

                - | sampler:Callable = A Callable handling MCMC
                    sampling. Possible values are `pymc.sample` for the
                    default `pymc` implementation  or
                    `sample_numpyro_nuts` for the faster `numpyro`/`jax`
                    implementation

                - | *args:typle[Any,...] := Arbitrary arguemnets to be
                    forwarded the sampler call

                - | **kwargs:dict[str,Any] := Arbitrary keyword
                    arguemnets to be forwarded to the sampler call

            Returns:
            ========

                - | idata:arviz.InferenceData := The inference data
                    container. Returned by the sampler call

            Updates the objects' :code:`_trained` and :code:`idata` attributes                
        '''
        self.__raise_uninitialized__()
        with self.model:
            self.idata = sampler(*args, **kwargs)
        self.__post_fit__()
        self._trained = True
        return self.idata

    
    def predict(self, Xnew, *args, verbosity:str = 'point_predictions',**kwargs):
        r'''
            Predict on new points, via the conditional distribution

            Args:
            =====

                - | Xnew:pandas.DataFrame := A DataFrame containing the
                    points to predict on

                - | *args:tuple := Arbitrary positional arguments to be
                    forwared to :code:`pymc.sample_posterior_predictive`

                - | **kwargs:dict[str,Any] := Arbitrary keyword
                    arguments to be forwared to
                    :code:`pymc.sample_posterior_predictive`
                
                - | verbosity:str='full_posterior' := Select if
                    common post processing operations are to be
                    performed on the posterior samples. Accepted values
                    are:

                    - 'full_trace' := Return unprocessed inference data
                      as an `xarray.DataSet`. Exactly what :code:`pymc.sample_posterior_predictive` returns

                    - | 'predictive_dist' := Basic prost processing
                        only. Stack the chain and draw dimensions and
                        return the `posterior_predictive` group only

                    - | 'point_predictions' := Return categorical labels
                        as they appeared in the unprocess dataframe (i.e
                        with their string labels)

            Returns:
            ========

            Data structure returned varies with the value of
            :code:`verbosity_level`

            - | trace:arviz.InferneceData := If
                :code:`verbosity_level='full_trace'` the full posterior
                predictive samples, as returned by
                :code:`pymc.sample_posterior_predictive`
            
            - | predictive_dist:xarray.DataSet := If
                :code:`verbosity_leve='predictive_dist'`. Return
                posterior predictive dataset with the `chain` and `draw`
                dimensions stacked into a new `sample` dimension.

            - | point_preds:pandas.DataFrame := If
                :code:`verbosity_level='point_predictions'`. Return a
                DataFrame with the predicted labels only. Indexers and
                labels are infered from the supplied training data

            Raises:
            =======

                - | RuntimeError := If the `_trained` or `_intialized`
                    sentinels are False
                    
                - | ValueError := If passed value of 'verbosity_level'
                    is not one of 'full_trace', predictive_dist or
                    'point_predictions'
                  
                  
        '''
        verbosity_choices:tuple[str,...] = (
            'full_trace', "predictive_dist", "point_predictions",
            "full_posterior"
            )
        self.__raise_any__()
        if verbosity not in verbosity_choices:
            raise ValueError((
                "Unknown verbosity argument. Expected on of 'full_trace', 'predictive_dist' or "
                f"'point_predictions' but received {verbosity} instead"
                ))
        with self.model:
            pm.set_data(dict(inputs=self.__preprocess_features__(Xnew, transform_only=True) ))
            self.trace = pm.sample_posterior_predictive(
                self.idata, *args, **kwargs
                )
        # BUG! When attempting to generate labels manually from the 
        # posterior predictive, (i.e. sample for the pymc.draw API) the
        # results are considerably worse (in terms of accuracy) than
        # letting the graph sample. This should be investigated further
        if verbosity == "full_trace" or verbosity=="full_posterior":
            return self.trace
        elif verbosity == 'predictive_dist':
            return self.trace.stack(sample=("chain", "draw")).posterior_predictive
        elif verbosity == 'point_predictions':
            
            a = self.trace.stack(
                sample=("chain", "draw")
                ).posterior_predictive['α_star'].mean(axis=-1).values
            labels = pymc.draw(
                    pymc.Categorical.dist(
                        a/a.sum(axis=-1, keepdims=True) )
                )
            return pd.DataFrame(
                numpy_replace(
                    labels.astype(str), self._decodings
                    )[:,None],
                columns=[self._target_label]
            )
