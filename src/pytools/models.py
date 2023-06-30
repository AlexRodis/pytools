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
from utilities import numpy_replace


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
    idata:typing.Optional[az.InferenceData] = field(repr=False, init =False, default=None)
    model:typing.Optional[pymc.Model] = field(repr=False, init =False, default=None)
    posterior:typing.Optional[az.InferenceData] = field(repr=False, init =False, default=None)
    coords:dict[str, typing.Any] = field(init=False, default_factory=dict)


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

                - RuntimeError := If the `_trained` or `_intialized`
                  sentinels are False
        '''
        self.__raise_untrained__()
        return az.plot_posterior(self.idata, *args, **kwargs)

    
    def plot_trace(self, *args, **kwargs):
        r'''
            Convenience wrapper for `arviz.plot_trace`

            Raises:
            =======

                - RuntimeError := If the `_trained` or `_intialized`
                  sentinels are False
        '''
        self.__raise_uninitialized__()
        return az.plot_trace(self.idata, *args, **kwargs)

    def summary(self, *args, **kwargs):
        r'''
            Convenience wrapper for `arviz.summary`

            Raises:
            =======

                - RuntimeError := If the `_trained` or `_intialized` sentinels are False
        '''
        return az.summary(self.idata, *args, **kwargs)

    def plot_energy(self, *args:tuple, **kwargs:dict[str,typing.Any]):
        r'''
            Plot MCMC transition energy

            Wrapper around :code:`arviz.plot_energy`. All arguements are forwarded
        '''

        return az.plot_energy(self.idata)

    def __pickle__(self):
        r'''
            Save the entire model with the `cloudpickle`
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
                    `arviz.InferenceData` object only. Optional. Defaults
                    to 'pickle'

            Raises:
            =======

                - | ValueError := If supplied method argumenet is not one
                    of 'pickle' or 'netcdf'
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

            - | method:str='pickle' := How the model was saved.
                If `method:str='pickle'` unpickles the entire object
                return a new instance. If `method:str='netcdf'`,
                creates a new instance with the specified inference data.
                When `method:str='netcdf'` the training data must also be
                supplied

            - | X_train:Optional[pandas.DataFrame=None] := Training features.
                Only required if `method:str='netcdf'`, otherwise ignored

            - | Y_train:Optional[pandas.DataFrame=None] := Training targets.
                Only required if `method:str='netcdf'`, otherwise ignored
                
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
                        \mathcal{GP}_0(\mu_0(\mathbf{x}), \kappa_0(\mathbf{x}))
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
                functionality to `sklearn.Pipeline` but only for
                preprocessing steps. Defines a series of objects with
                `fit_transform` and `transform` methods, for data
                preprocessing (i.e rescalling, dimensionality reduction,
                etc)

            - | features := A `numpy.typing.NDArray` containing labels
                for the input features. Extracted from passed data

            - | classes := A `numpy.typing.NDArray` containing unique
                classes to predict

            - | model:Optional[pymc.Model] := A reference to the
                `pymc.Model` object

            - | idata:Optional[arviz.InferenceData] := Posterior samples
                from MCMC inference

            - | posterior:Optional[arviz.InferenceData] := Posterior
                predictive samples. Stores the values from the most
                recent call to `predict`

            - | approximate:bool=True := If :code:`True` use the Hilbert 
                Space approximation for faster inference. Only effective 
                for up to 3-4 input dimensions. Optional. Defaults to
                :code:`True`.

            - | hsgp_kwargs:dict[str,Any] := Keyword arguments to be
                forwarded to `pymc.gp.HSGP`. Generally  hyperparameters
                for the approximation

            - | perturbation_factor:float=1e-6 := A small perturbation
                factor for observed points. Since the likelihood of the
                Dirichlet is :math:`+\infty` at simplex edges, we shift
                the observations but this small factor to improved
                numerical stability

            - | save_dir:Optional[str] := Directory to save the model
                to. Optional

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
                `pymc.go.HSGP` when `approximate=True` or
                `pymg.gp.Latent` when `approximate=False` (untested)

            - | _initialized:bool=False := Model specification sentinel.
                Prevents calls to post-__call__ methods until the
                `__call__` method has specified the model

            - | _trained:bool=False := Model inference sentinel.
                Prevents calls to post-fit methods until the `fit`
                method has been called

        Object Public Methods:
        ========================

            - | __init__(pipeline:Pipeline)->self. := An instance of 
                `Pipeline` encapsulating data preprocessing operations

            - | __call__(X:pandas.DataFrame,y:pandas.DataFrame)->self := 
                Specify the full probability model for inference, providing
                inputs :code:`X` and outputs :code:`y`

            - | fit(sampler:Callable=pymc.sample, *sampler_args:tuple, 
                **sampler_kwargs:dict[str,Any])->arviz.InferenceData := 
                Perform inference on the data, using MCMC. Returns a
                :code:`arviz.InferenceData` object containing the results
                of infernce. :code:`sampler` is a Callable specifying which
                implementation of MCMC to use. Defaults to :code:`pymc.sample`.
                Other options include :code:`pymc.sample_numpyro_nuts`, which
                required external dependencies

            - | predict(Xnew:pandas.DataFrame, *args:tuple, verbosity_level:int=2,
                kwargs:dict[str,Any])->arviz.InferenceData := Predict on new points 
                :code:`Xnew`, with optional postprocessing logic defined by the 
                :code:`verbosity_level` arguement. For `verbosity_level=0`,
                returns point predictions are labels, inference from the training data.
                For :code:`verbosity_level=1` sample concentration, and stack the chain 
                and draw dimensions into a new dimension named :code:`samples`. For
                :code:`verbosity_level=2` returns the full posterior predictive samples.
                All other arguements are forwarded to 
                :code:`pymc.sample_posterior_predictive`.

            - | save(save_dir:Optional[str]=None, method:str='pickle')->None := Save 
                the model into target directory. :code:`save_dir` must be provided, 
                either as an object attribute or during the call. For 
                code:`method='pickle'` will serialize the entite object for 
                ready reuse. For :code:`method=netcdf` only saves the posterior 
                code:`InferenceData` object. When loading the latter from memory no 
                checks are being made that the posterior trace is consistant with the 
                model object.

            - | load(save_dir:str, method:str='pickle')->DirichletGPClassifier := Load
                a model from memory according to the method specified by :code:`method`.
                For :code:`method='pickle'` simply unserializes the object and returns
                a the new instance of :code:`DirichletGPClassifier`

            - | plot_trace(*args:tuple, **kwargs:dict)-> := Display a trace plot of
                model inference. Wrapper for :code:`arviz.plot_trace`. 
                All arguements are forwarded.             

            - | plot_energy(*args:tuple, **kwargs:dict)->pandas.DataFrame := Display an
                energy plot for model inference. Wrapper for :code:`arviz.plot_energy`. 
                All arguments are forwarded.

            - | plot_posterior(*args:tuple, **kwargs:dict)-> := Display a 
                plot of the models posterior. Wrapper for :code:`arviz.plot_posterior`. 
                All arguements are forwarded.            
    '''
    pipeline:Pipeline = field(default= None)
    features:typing.Optional = field(init =False, default=None)
    classes:typing.Optional = field(init =False, default=None)
    model:typing.Optional[pymc.Model] = field(repr=False, init =False, default=None)
    idata:typing.Optional[az.InferenceData] = field(repr=False, init =False, default=None)
    posterior:typing.Optional[az.InferenceData] = field(repr=False, init =False, default=None)
    approximate:bool=True
    hsgp_kwargs:dict = field(default_factory=dict)
    perturbation_factor:float= 1e-6
    trace:typing.Optional[az.InferenceData] = field(init=False, default=None)
    save_dir:typing.Optional[str] = None
    
    _encodings:typing.Optional[dict] = None
    _decodings:typing.Optional[dict] = None
    _n_features:typing.Optional[int] = field(init = False, default = None)
    _n_inputs:typing.Optional[int] = field(init = False, default = None)
    _n_obs:typing.Optional[int] = field(init = False, default = None)
    _n_classes:typing.Optional[int] = field(init = False, default = None)
    _classes:typing.Optional = field(default = None)
    _means:typing.Optional[np.typing.NDArray] = None
    _target_label:typing.Optional[str] = None
    _coords:typing.Optional[dict] = field(repr=False, init=False, default_factory=dict)
    _processor:typing.Optional = field(init=False, repr=False)
    _prior_latents:typing.Optional[list] = field(repr= False, init=False, default_factory=list)
    _conditional_latents:typing.Optional[list] = field(repr= False, init=False, default_factory=list)
    _processes:typing.Optional[list] = field(repr= False, init=False, default_factory=list)
    _initialized:bool = False
    _trained:bool = False
    _3d_cache:typing.Optional=field(init=False, repr=False, default=None)
    _default_colors:typing.Optional[list[str]] = field(repr=False, init=False,
        default=None)
    _linspace:typing.Optional[typing.NamedTuple] = field(repr=False, init=False, default=None) 

    def __post_init__(self)->None:
        r'''
            Initialize the gaussian processor
        '''
        from collections import namedtuple
        if self.approximate:
            self._processor = pymc.gp.HSGP
        else:
            self._processor = pymc.gp.Latent

        self._default_colors= ['plasma','viridis','blues','aggrnyl', 'agsunset', 'algae', 
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
                 'ylorrd']

        self._linspace = namedtuple('LinearSpace', ['start', 'stop', 'n_points'])
        

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
            for id in self._classes:
                ℓ = pymc.Normal(f'ℓ_{id}', mu = 7., sigma=1.5, shape=(M,))
                κ_se = pymc.gp.cov.ExpQuad(M, ls=ℓ )
                κ = κ_se
                C = 0.0
                μ = pymc.gp.mean.Constant(c=C)
                gp = self._processor(
                    mean_func=μ, cov_func=κ, c=1.3,m=[7]*M )
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

            Constructs additional nodes in the computation graph for
            predictions on new points. Create a data note for unseen
            points and duplicates all GP nodes to generate the
            corresponding conditional distributions.

            Conditional variables are always named with suffix "_star",
            for example "f_star", "α_star" and "y_star".
        '''
        with self.model:
            inputs = pymc.MutableData('inputs', np.random.rand(3,self._n_inputs))
            for id,gp in zip(self._classes,self._processes):
                f_star=gp.conditional(f'f_star_{id}',inputs)
                self._conditional_latents.append(f_star)
            f_star = pytensor.tensor.stack(self._conditional_latents).T
            α_star = pymc.Deterministic('α_star', pymc.math.exp(f_star) )
            y_star = pymc.Dirichlet('y_star', a = α_star)
            

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

            Updates the objects' `_trained` and `idata` attributes
                    

                
        '''
        self.__raise_uninitialized__()
        with self.model:
            self.idata = sampler(*args, **kwargs)
        self.__post_fit__()
        self._trained = True
        return self.idata

    
    def predict(self, Xnew, *args, verbosity_level:int=0,**kwargs):
        r'''
            Predict on new points, via the conditional distribution

            Args:
            =====

                - | Xnew:pandas.DataFrame := A DataFrame containing the
                    points to predict on

                - | *args:tuple := Arbitrary positional arguments to be
                    forwared to `pymc.sample_posterior_predictive`

                - | **kwargs:dict[str,Any] := Arbitrary keyword
                    arguments to be forwared to
                    `pymc.sample_posterior_predictive`
                
                - | verbosity_level:int=0 := Select if common post
                    processing tasks are to be performed on the
                    posterior samples. Acceptable levels are:

                    - 2 := Return unprocessed inference data as an
                      `xarray.DataSet`

                    - | 1 := Stack the chain and draw dimensions and
                        return the `posterior_predictive` group only

                    - | 0 := Return categorical labels as they appeared
                        in the unprocess dataframe (i.e with their
                        string labels)

            Returns:
            ========

            Data structure returned varies with the value of
            `verbosity_level`

            - | preds:arviz.InferneceData := If `verbosity_level=2`.
                Full posterior predictive samples, as returned by
                `pymc.sample_posterior_predictive`
            
            - | preds:xarray.DataSet := If `verbosity_leve=1`. Return
                posterior predictive dataset with the `chain` and `draw`
                dimensions stacked into a new `sample` dimension

            - | preds:pandas.DataFrame := If `verbosity_level=0`. Return
                a DataFrame with the predicted labels only. Indexers and
                labels are infered from the supplied training data

            Raises:
            =======

                - RuntimeError := If the `_trained` or `_intialized`
                  sentinels are False
        '''
        self.__raise_any__()
        with self.model:
            pm.set_data(dict(inputs=self.__preprocess_features__(Xnew, transform_only=True) ))
            self.trace = pm.sample_posterior_predictive(self.idata, *args, **kwargs)
        if verbosity_level == 2:
            return self.trace
        elif verbosity_level == 1:
            return self.trace.stack(sample=("chain", "draw")).posterior_predictive
        elif verbosity_level == 0:
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
