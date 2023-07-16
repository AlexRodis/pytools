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

from dataclasses import dataclass, field
from .typing import NDArray
import typing
import numpy as np
import pandas as pd
import screeninfo
import plotly
import plotly.graph_objects  as go
import seaborn as sns
import matplotlib.pyplot as plt

@dataclass(slots=True, kw_only=True)
class SurfaceVisualizer:
    r'''
        Parent class for multi dimensional model visualizations.


        Object Public Attributes:
        ==========================

            - | model:pycm.Model := The model object

            - | var_name:str="α_star" := Alias of the variable to
                visualize. Optional. Defaults to :code:`"α_star"`
                for the :code:`DirichletGPClassifier`

            - | grid_space:tuple[tuple[float,float,int],
                tuple[float,float,int]]=((0,1,50)(0,1,50)) := The
                space over which to generate a meshgrid for model
                evaluations. Is a length 2 tuple, whose elements
                are length 3 tuples, specifying linear spaces for
                for a pair of features. Linear spaces are given
                in the form :code:`(start:float,stop:float, n_points:int)`.
                :code:`n_points` must be a positive integer. Optional.
                Defaults to a uniform grid in the space 
                :math:`[0,1]\times [0,1]` with size :math:`50\times 50`.

            - | feature_labels:Sequence[str] := A sequence of strings
                that are coordinates / labels for input features. Length
                must exactly match the provided data's second axis

            - | predictor_labels:Sequence[str] := A sequence of strings,
                specifying coordinate labels for model output variable.
                Length must exactly match the second axis of the variable
                specified

            - | placeholder_vals:np.typing.NDArray[float] := Values used
                to pad the grid features other than those depicted. Possible
                use cases are zeros of the mean of the feature. Must be a
                vector whose length match the second dimension of the data

            - | colormaps:Optional[Sequence[str]] := A sequence of strings
                which are named plotly color scales to be used in the 
                visualization. Any missing values will set to defaults. Any
                additional values are ignored

            - | smoothing:Optional[list[float,float]]=None := A length 2 
                sequence of floats, specifying the gaussian smoothing
                parameters 'sigma'. Optional. If :code:`None`, no smoothing
                is applied. Otherwise, gaussian smoothing with the specified
                sigmas is applied

            - | scaling_factor:float=.8 := Plot size scaling factor. Relative
                to user display. Optional and defaults to .8

            - | colorbar_spacing_factor:float=.05 := Factor that controls the
                spacing between the plots' colorbars. Optional and defaults to
                .05

            - | colorbar_location:float=1. := Location for colorbar placement.
                Optional and defaults to 1

            - | adaptable_z_axis:bool=True := Controls is the size if of the z
                axis should be set according to data. If :code:`False`, axis
                is given proability boundaries at [0,1] regardless of the data

            - | autoshow:bool=False := If :code:`True` automatically call the
                :code:`plotly.Figure` object's :code:`show` method to render
                the figure

            - | layout:dict[str,Any] := A plot layout dictionary. Optional.
                Allows overriding of default titles, names etc
        
    '''
    model :typing.Optional[typing.Any] = field(default=None)
    var_name:str = field(default = "α_star")
    grid_space:tuple[tuple[float,float,int], 
                     tuple[float,float,int]] = field(
                        default =  ((0,1,50), (0,1,50))
                     )
    grid:tuple[tuple[float,float, int],tuple[float,float,int]]= field(
        default_factory=tuple
    )
    feature_labels:typing.Optional[list[str]] = field(
        default = None
    )
    predictor_labels:typing.Optional[list[str]] = field(default=None)
    placeholder_vals:typing.Optional[np.typing.NDArray] = field(
        default = None)
    colormaps:typing.Sequence[str] = field(default_factory=list)
    smoothing:typing.Optional[typing.Sequence[float]] = None
    scaling_factor:float = field(default=.8)
    colorbar_spacing_factor:float = .05
    colorbar_location: float = 1
    layout:typing.Optional[dict[str,typing.Any]] = field(
        default_factory = dict)
    adaptable_zaxis :bool = True
    autoshow :bool = False
    
    _default_colors:typing.Optional[list[str]] = field(
        repr=False, init=False, default=None
    )
    _3d_cache:typing.Optional[typing.NamedTuple] = field(
        repr=False, init=False, default=None
    )
    _space_factory:typing.Optional[typing.NamedTuple] = field(
        repr=False, init=False, default=None
    )
    _trace_cache_factory:typing.Optional[typing.NamedTuple] = field(
        repr=False, init=False, default=None
    )
    _n_features:typing.Optional[int] = None
    
    def __post_init__(self)->None:
        from typing import NamedTuple
        from arviz import InferenceData
        
        class LinearSpace(NamedTuple):
            start :float = .0
            stop :float = 1.0
            n_points :int = 50
        
        self._space_factory = LinearSpace
        
        self._default_colors = ['plasma','viridis','blues','aggrnyl', 'agsunset', 'algae', 
        'amp', 'armyrose', 'balance',
        'blackbody', 'bluered', 'blugrn', 'bluyl', 'brbg',
        'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
        'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 
        'electric', 'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys', 
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
        
        class TraceCache(NamedTuple):
            trace :InferenceData = None
            x_feature :typing.Optional[str] = None
            y_feature :typing.Optional[str] = None
            
        self._trace_cache_factory = TraceCache
        if self.predictor_labels is None:
            raise ValueError((
                "`pred_labels` argument must be provided as a list of "
                "string labels for mapping the columns of model "
                f"predictions. Received {self.pred_labels} instead"
            ))
        self._n_features = len(self.predictor_labels)
            
    def __get_screen__(self)->tuple[int,int]:
        r'''
            Fetch screen width and height for plot scaling

            Args:
            =====

                - | scaling_factor:float := A factor to scale the screen
                    by for plotting

            Returns:
            =========

                - | width:int := Scaled screen width

                - | height:int := Scaled screen height
            
        '''
        from screeninfo import get_monitors
        monitor = get_monitors()[0]
        width:int = int(monitor.width*self.scaling_factor)
        height:int = int(monitor.height*self.scaling_factor)

        return width, height

    def __extend_colormaps__(self, required:int)->list[str]:
        r'''
            DRY helper method for colorscale preprocessing. If
            user has not explicitly all the colormaps required,
            pads the supplied list with defaults for any that
            are missing

            Args:
            ======

                - | required:int := The number of colormaps required.
                    Is the number of surfaces to depict

            
            Returns:
            =========

                - | new_cmap:list[str] := A new list of named colormaps
                    to use. fewer than required were supplied, extras
                    from defaults are used
        '''
        _colors:list[str] = self.colormaps
        _n_missing:int = required - len(_colors)               
        if _n_missing>0:
            _colors.extend(self._default_colors[:_n_missing])
        return _colors

    def __prep_meshgrid__(self, spaces:tuple[tuple[float,float,int]], 
                          inputs:tuple[str,str],
                          targets:typing.Sequence[str]
                          )->tuple[typing.NamedTuple, np.typing.NDArray[float],
                                    np.typing.NDArray[float], np.typing.NDArray[float],
                                    list[int]]:
        r'''
            DRY helper method for meshgrid creation

            Args:
            =====

                - | space:tuple[float,float,int] := A triplet of values
                    defining the 1D linear space used to generate a
                    uniform grid. Of the form (start,stop,n_points)

                - | inputs:tuple[str,str] := A pair of feature
                    labels/coordinates to use in making the meshgrid.
                    All others will be set to the feature mean

                - | targets:Sequence[str] := Class labels to include in
                    the plot


            Returns:
            =========

                - | linspace:namedtuple[start:float,stop:float,
                    n_points:int := The space packaged as a namedtuple
                    with the fields 'start', 'stop' and 'n_points'

                - | xaxis:numpy.NDArray[float] := The linear space as a
                    vector of floats for one input feature

                - | yaxis:numpy.NDArray[float] := The linear space as a
                    vector of floats for the other input feature

                - | zarr:numpy.NDArray[float] := A meshgrid for function
                    evaluations in the space defined by xaxis and yaxis

                - | target_indices:list[int] := A list of integer
                    indices mapping to the specified target labels

        '''
        linspaces = [
            self._linspace(
                start=space[0], stop=space[1], n_points=space[2]
                ) for space in spaces
        ]
        replace_targets = (
            self._coords['features'].to_list().index(inputs[0]), 
            self._coords['features'].to_list().index(inputs[1])
        )
        target_labels_to_indices = [
            self._classes.index(t) for t in targets
            ]
        xaxis, yaxis, zarr = self.__create_meshgrid__(
            linspaces, replace_targets=replace_targets
            )

        return linspaces, xaxis, yaxis, zarr, target_labels_to_indices

    def __eval_model__(self,
                       features:tuple[str,str], 
                       zarr:np.typing.NDArray[float]
                       )->pd.DataFrame:
        r'''
            DRY helper method for model evaluations over a specified
            grid


            Note:
            
            To avoid uneccessary grid evaluations, we save the last
            generated trace in the `_3d_cache` attribute and only call
            the models' predict if the input features change

            Args:
            ======

                - | inputs:tuple[str,str] := A pair of feature
                    labels/coordinates to use in making the meshgrid.
                    All others will be set to the feature mean

                - | zarr:numpy.NDArray[float] := The meshgrid to
                    evaluate the model over. Assumed to be reshaped to a
                    form suitable for model evaluation

            Returns:
            ========

                - | probs:pandas.DataFrame := An N x K DataFrame
                    containing predicted proability distributions for K
                    classes over N points. N is the product of the
                    number of points across all features (here always
                    2). Columns are named according the scheme
                    "$\hat{p}_{class_label}$"


            
        '''
        c_undefined:bool = self._3d_cache is None
        if c_undefined:
            trace = self.model.predict(
                zarr, var_names=[self.var_name], 
                verbosity_level=2
                )
            self._3d_cache = self._trace_cache_factory(
                x_feature = features[0], y_feature = features[1],
                trace = trace
                )
        else:
            c_feature_change:bool = self._3d_cache.x_feature != \
                    features[0]
            c_other_feature_change:bool = self._3d_cache.y_feature != \
                features[1]
            if c_feature_change or c_other_feature_change:
                trace = self.model.predict(
                zarr, var_names=[self.var_name],verbosity_level=2
                )
                self._3d_cache = self._trace_cache_factory(
                x_feature = features[0], 
                y_feature = features[1],
                trace = trace
                )
        trace = self._3d_cache.trace
        
        p=trace.posterior_predictive.stack(
            sample=('chain', 'draw')
            )[self.var_name].mean(axis=-1).values
        probs=pd.DataFrame(
            p/p.sum(axis=1,keepdims=True), 
            columns = self.predictor_labels
            )
        return probs

    def __update_layout__(self,
                          title:str, 
                          usr_layout:dict,
                          width:int,
                          height:int )->plotly.graph_objects.Figure:
        r'''
            DRY helper method for plot adding plot details
        '''
        
        base_layout = dict(
        title=title,
        autosize=True,
        width=width,
        height=height,
        )
        fig.update_layout(base_layout)
        fig.update_layout(usr_layout)
        return fig
    
    def __prep_meshgrid__(self,
                          spaces:tuple[tuple[float,float,int]], 
                          features:tuple[str,str],
                          targets:typing.Sequence[str]
                          )->tuple[
                              typing.NamedTuple, np.typing.NDArray[float],
                            np.typing.NDArray[float], np.typing.NDArray[float],
                                    list[int]]:
        r'''
            DRY helper method for meshgrid creation

            Args:
            =====

                - | space:tuple[float,float,int] := A triplet of values
                    defining the 1D linear space used to generate a
                    uniform grid. Of the form (start,stop,n_points)

                - | inputs:tuple[str,str] := A pair of feature
                    labels/coordinates to use in making the meshgrid.
                    All others will be set to the feature mean

                - | targets:Sequence[str] := Class labels to include in
                    the plot


            Returns:
            =========

                - | linspace:namedtuple[start:float,stop:float,
                    n_points:int := The space packaged as a namedtuple
                    with the fields 'start', 'stop' and 'n_points'

                - | xaxis:numpy.NDArray[float] := The linear space as a
                    vector of floats for one input feature

                - | yaxis:numpy.NDArray[float] := The linear space as a
                    vector of floats for the other input feature

                - | zarr:numpy.NDArray[float] := A meshgrid for function
                    evaluations in the space defined by xaxis and yaxis

                - | target_indices:list[int] := A list of integer
                    indices mapping to the specified target labels

        '''
        linspaces:list[self._space_factory] = [
            self._space_factory(
                start=space[0], stop=space[1], n_points=space[2]
                ) for space in spaces
        ]
        self.grid=linspaces
        # TODO Refactor this
        replace_targets = (
            self.feature_labels.to_list(
                ).index(features[0]), 
            self.feature_labels.to_list(
                ).index(features[1])
        )
        # TODO Refactor this
        target_labels_to_indices = [
            self.model._classes.index(t) for t in targets
            ]
        xaxis, yaxis, zarr = self.__create_meshgrid__(
            replace_targets=replace_targets
            )
        return linspaces, xaxis, yaxis, zarr, target_labels_to_indices
    
    def __precall_checks__(self,
                           targets:typing.Sequence[str],
                           features:tuple[str,str])->None:
        
        if targets is None:
            raise ValueError()
        if features is None:
            raise ValueError()
        if len(features) != 2:
            raise ValueError()
        
    def __create_meshgrid__(self, 
                            replace_targets:tuple[int,int]=(0,1), 
                            )->tuple[np.typing.NDArray,
                                    np.typing.NDArray,
                                    pd.DataFrame]:
        r'''
            Create a meshgrid for multivariate function evaluation. 

            Args:
            ======

                - | spaces:Sequence[namedtuple,namedtuple] := A pair of
                    namedtuples with the fields 'start', 'stop',
                    'n_points', each defining the linear space for the
                    grid to generate

                - | replace_targets:tuple[int,int]=(0,1) := Which
                    feature indices to create a space for. All others
                    are set to their means

            Returns:
            ========

                - | xaxis:numpy.typing.NDArray := A numpy vector
                    representing the linear space (start, stop, end)

                - | yaxis:numpy.typing.NDArray := A numpy vector
                    representing the linear space (start, stop, end)

                - | grid_matrix:pandas.DataFrame : = The grid in the
                    form of a 2D matrix `pandas.DataFrame`, suitable for
                    insertion into the model

        '''
        aspace, bspace = self.grid
        xa = np.linspace(aspace.start, aspace.stop, num=aspace.n_points)
        xb = np.linspace(bspace.start, bspace.stop, num=bspace.n_points) 
        _x,_y = np.meshgrid(xa,xb)
        base_grid = np.stack((_x.flatten(), _y.flatten()), axis=1)
        N, M = base_grid.shape[0], self._n_features
        means = self.placeholder_vals
        husk = np.tile(means, (N,1))
        husk[:,replace_targets[0]] = base_grid[:,0]
        husk[:,replace_targets[1]] = base_grid[:,1]
        return xa, xb, pd.DataFrame(husk,
                                    columns= self.feature_labels)
        
    def __call__(self, 
                 targets:typing.Optional[typing.Sequence[str]] = None, 
                 features:typing.Optional[tuple[str,str]] = None
                )->go.Figure:
        r'''
            Plot 3D visualization of class probability surface for
            against a pair of input features

            Args:
            =====

                - | targets:Sequence[str] := A sequence of class labels
                    to plot

                
                - | inputs:tuple[str,str] := A pair of feature labels to
                    plot

                
                - | space:tuple[int,int,int] := A triplet of integers
                    defining a the linear space used to construct the
                    grid. Of the form (start,stop,n_points)
                
                - | smooth:Optional[Sequence[int,int]]=None := Gaussian
                    smoothing sigmas. Set to `None` to disable surface
                    smoothing. Optional. Defaults to None.

                - | scaling_factor:float=.8 := General plot scaling
                    factor. Relative to screen size (in pixels).
                    Optional. Defaults to .8.

                - | autoshow:bool=False := Select wheather of not to
                    automatically call the `show` method on the figure.
                    Optional. Defaults to `False`.

                - | layout:dict=dict() := Extra layout arguements
                    forwarded to the call to `update_layout()`. Default
                    options can be overriden with user specifications

                - | cmap:list[str]=list() := Named colormaps for the
                    surface. Any missing arguements are replaced with
                    defaults. Optional

                - | colorbar_spacing_factor:float=.05 := Set extra blank
                    space between multiple colorbars

                - | adaptable_zaxis:bool=False := Select if the z axis
                    should have a fixed range of values (in the [0,1]
                    space) to allow easier comparisons between plots or
                    be adapted based on the probability density.
                    Optional. Defaults to False and and fixes the axis
                    in the [0,1] interval.

                - | colorbar_location:float=1 := Location where
                    colorbars will be placed. Optional. Defaults to 1
                    (right hand side of the graph). .5 is in the middle
                    of the graph

            Returns:
            =========

                - | figure:plotly.graph_object.Figure := The `plotly`
                    figure object

        '''
        self.__precall_checks__(targets, features)
        width, height = self.__get_screen__()
        _ = self.__prep_meshgrid__(self.grid, features, targets)
        linspaces = _[0]
        xaxis = _[1]
        yaxis = _[2]
        zarr = _[3]
        target_labels_to_indices = _[4]
        preds :pd.DatFrame = self.__eval_model__(features, zarr)
        fig = self.__create_3d_plot__(
            xaxis, yaxis, zarr,preds ,
            pred_idxs=target_labels_to_indices, 
            )
        scene = dict(
                xaxis_title = features[0],
                yaxis_title = features[1],
                zaxis_title = 'Probability',                
            )
        space_string_l:str = f"[{linspaces[0].start},{linspaces[0].stop}]"
        space_string_r:str = f"[{linspaces[1].start},{linspaces[1].stop}]"
        title:str = r'$\text{Response Surface}\ '+space_string_l + \
            "\ \\times\ " + space_string_r+"$"
        base_layout = dict(
            title = title,
            scene = scene,
            autosize=True,
            width = width,
            height = height,
            margin = dict(l=0, r=80, b=35, t=30),
        )
        fig.update_layout(
            **base_layout
        )
        # Allow user specified settings to override the defaults
        fig.update_layout(**self.layout)
        if self.autoshow:
            fig.show()
        return fig

    def __create_3d_plot__(*args, **kwargs)->None:
        raise NotImplementedError()

@dataclass(slots=True, kw_only=True)
class ResponseSurfaceVisualizer(SurfaceVisualizer):
    r'''
        Visualize multidimensional model variables by creating
        a response surface plot of the target variable against
        a pair of input features
        
        Example usage:
        
        .. code:: Python 3
        
            from pytools.visualizations import ResponseSurfaceVisualizer
        
            vizer = ResponseSurfaceVisualizer(
                    model = obj, 
                    var_name = "α_star",
                    smoothing = [2,2],
                    grid = ((0,50,50),(0,50,50) ),
                    placeholder_vals = X.mean(axis=0),
                    feature_labels=  ["Feature 1", "Feature 2"],
                    predictor_labels = [0,1,2],
                    colormaps = [
                        'viridis', 'magma', 'tealrose', 
                        "inferno", "blues"
                        ]
                    scaling_factor = .8,
                    colorbar_spacing_factor= .05,
                    colorbar_location = .8,
                    layout = dict(),
                    adaptable_zaxis = False,
                    autoshow=False
                )
                
        Object Public Attributes:
        ===========================
        
            - | model:pytools.BayesianModel := The trained model to
                visualize
            
            - | var_name:str := The name of the variable to plot as a 
                response. Optional and defaults to "α_star".
                
            - | smoothing:Optional[[int,int]]=None := Selects the sigmas
                for gaussian filter smoothing. If :code:`None` no
                smoothing will be applied. Otherwise a supply a length-2
                list of positive integers, which are the sigmas for the
                filter
                
            - | grid:Iterable[Iterable[int,int,int],
                Iterable[int,int,int]] := Specification for the
                evaluation grid to generate. Is a length 2
                :code:`Iterable` whose elements correspond to the two dimensions on the grid. For each dimension a length 3
                :code:`Sequence` must be supplied of the format 
                :code:`start, stop, n_points`.
                
            - | placeholder_vars:numpy.NDArray := A feature length
                vector corresponding to the fixed values which all other
                features, other than the ones plotted will be fixed to.
            
            - | feature_labels:Iterable[str,str] := A length-2
                :code:`Iterable` with the labels for two feature
                dimensions
                  
            - | predictor_labels:Sequence := A sequence of labels for
                surfaces to plot
                
            - | colormaps:Iterable[str]:= A sequence of named
                :code:`plotly` colorscales to be used for plotting
                surfaces. Any missing ones will be replaced by defaults,
                while any excess ones will be ignored. Optional
                
            - | scaling_factor:float=.8 := General rescaling factor,
                relative to user monitor. Optional and defaults to 1
                (full screen). Optional.
                
            - | colorbar_spacing_factor:float := Controls the whitespace
                between the response surfaces colorbars
                
            - | colorbar_location:float := Shift location of the
                colorbars as defined by :code:`plotly`. Optional.
                
            - | layout:dict := Additional plot layout arguments to be
                forwarded to
                :code:`ploty.graph_objects.Figure.update_layout`.
                Optional. Defaults to an empty dict.
            
            - | adaptable_zaxis:bool=False := Control z axis range. If
                :code:`True` the range of the axis will be infered from
                the data. If :code:`False` the axis range will be set to
                [0,1] regardless of the data. Optional. Defaults to :code:`False`
                
                            
            - autoshow:bool=False := If :code:`True` automatically calls
              the plotly method :code:`plotly.graph_objects.Figure.show`
              method. Optional. Defaults to :code:`False`.
    '''

    def __create_3d_plot__(self,
                           xaxis:np.typing.NDArray, 
                           yaxis:np.typing.NDArray, 
                           zgrid:np.typing.NDArray, 
                           preds:pd.DataFrame,
                           pred_idxs:typing.Sequence[int]=[0], 
                           ):
        r'''
            Create a 3D surface plot

            Args:
            ======

                
                - | xaxis:numpy.typing.NDArray := Array of the x axis
                    linear space
                
                - | yaxis:numpy.typing.NDArray := Array of the y axis
                    linear space
                
                - | zgrid:numpy.typing.NDArray := 2D Array containing z
                    coordinates for the surface. Must be of exact shape
                    |xaxis| x |yaxis|.

                - | preds:pandas.DataFrame := DataFrame containing model
                    evaluations for the specified |xaxis| x |yaxis| grid

                - | class_idx:Sequence[int]=[0] := Indices for the
                    dimensions of model output to plot.

            Returns:
            =========

                - | figure:ploty.graph_objects.Figure := The generated
                    3D surface plot


        '''
        from scipy.ndimage import gaussian_filter

        sh_0, sh_1 = xaxis.shape[0], yaxis.shape[0]
        _colors = self.__extend_colormaps__(len(pred_idxs))
        fig = go.Figure()
        for i, (idx, colormap) in enumerate(zip(pred_idxs, _colors)):
            z_grid = preds.values[:,idx].reshape(sh_0, sh_1)
            if self.smoothing is not None:
                z_smooth = gaussian_filter(
                    z_grid, self.smoothing
                    )
            else:
                z_smooth=z_grid
            c_loc:float = self.colorbar_location
            c_space:float = self.colorbar_spacing_factor
            x_loc:float = c_loc+i*c_space
            kwargs = dict(
                    x=xaxis, y= yaxis, z = z_smooth, 
                    colorscale = colormap,
                    colorbar = dict(
                        x = x_loc, 
                    title = self.predictor_labels[idx]), 
                )
            if not self.adaptable_zaxis:
                kwargs["cmin"] = 0
                kwargs['cmax'] = 1
            fig.add_trace(
                go.Surface(
                    **kwargs                   
                )
            )
        return fig
                               
@dataclass(slots=True, kw_only=True)
class ContourSurfaceVisualizer(SurfaceVisualizer):
    r'''
        Create a contour plot visualization for multidimensional
        model variables.
    '''

    opacity:float = 1.
    
    def __create_3d_plot__(self,
                           xaxis:np.typing.NDArray, 
                           yaxis:np.typing.NDArray, 
                           zgrid:np.typing.NDArray, 
                           preds:pd.DataFrame,
                           pred_idxs:typing.Sequence[int]=[0], 
                           ):
        r'''
            Create a 3D surface plot

            Args:
            ======

                
                - | xaxis:numpy.typing.NDArray := Array of the x axis
                    linear space
                
                - | yaxis:numpy.typing.NDArray := Array of the y axis
                    linear space
                
                - | zgrid:numpy.typing.NDArray := 2D Array containing z
                    coordinates for the surface. Must be of exact shape
                    |xaxis| x |yaxis|.

                - | preds:pandas.DataFrame := DataFrame containing model
                    evaluations for the specified |xaxis| x |yaxis| grid

                - | class_idx:Sequence[int]=[0] := Indices for the
                    dimensions of model output to plot.

            Returns:
            =========

                - | figure:ploty.graph_objects.Figure := The generated
                    3D surface plot


        '''
        from scipy.ndimage import gaussian_filter

        sh_0, sh_1 = xaxis.shape[0], yaxis.shape[0]
        _colors = self.__extend_colormaps__(len(pred_idxs))
        fig = go.Figure()
        for i, (idx, colormap) in enumerate(zip(pred_idxs, _colors)):
            z_grid = preds.values[:,idx].reshape(sh_0, sh_1)
            if self.smoothing is not None:
                z_smooth = gaussian_filter(
                    z_grid, self.smoothing
                    )
            else:
                z_smooth=z_grid
            c_loc:float = self.colorbar_location
            c_space:float = self.colorbar_spacing_factor
            x_loc:float = c_loc+i*c_space
            kwargs:dict = dict(
                x=xaxis, y= yaxis, z = z_smooth, 
                colorscale = colormap, opacity=self.opacity,
                colorbar = dict(
                        x = x_loc, 
                    title = self.predictor_labels[idx]),           
                )
            if self.adaptable_zaxis:
                kwargs['zmin'] = 0
                kwargs['zmax'] = 1
            fig.add_trace(
                go.Contour(
                     **kwargs   
                ))
        return fig


def categorical_scatter(X:pd.DataFrame, Y:typing.Optional[pd.DataFrame],
    cols:int=3, max_rows:int = 3, figsize:tuple[int, int] = (30, 15),
    xaxis_label_size: int=12, yaxis_label_size:int=12,
    categorical:typing.Optional[str]='hue')->typing.Optional[plt.figure]:
    '''
        Generate pair-wise scatter plots for a given DataFrame, with optional 
        support for large Datasets and categorical variables. Yields a 
        figure with `max_rows x cols` scatterplots per call. When 
        `categorical=None` only generates pair-wise scatterplots for the 
        columns of X, else generates all combinations of X-column pair and Y 
        columns. When `categorical='hue'` values of the categorical Y variable 
        will be depicted color-coded and when `categorical=size` different 
        factors of the Y variable will have different sizes instead.
        
        Args:
        ======
        
            - | X:pandas.DataFrame := The data to depict. When Y is also
            specified, X is assumed to be the DataFrame of indicator
            variables
            
            - | Y:Optional[pandas.DataFrame] := Optional Dataframe of
            categorical variables to depict. Ignored if `category` is
            `None`.
            
            - | idx_lvl:int=1 := For multilevel indexed dataframes the
            lowest level to squash the index on. Must be non-negative
            
            - | cols:int=3 := Number of columns for the resulting facet
            grid plot. Must be non-negative. Defaults to 3
            
            - | max_rows:int=3 := Maximum number of rows per batch the
            generator yields. Defaults to 3 and must be non-negative
            
            - | figsize:tuple[int, int] := A `height x width` tuple for
            the generated plots. Defaults to `(30, 15)`
            
            - | xaxis_label_size:int=12 := The size of the x-axis titles
            for each subplot. Must be non-negative and defaults to 12
            
            - | yaxis_label_size:int=12 := The size of the y-axis titles
            for each subplot. Must be non-negative and defaults to 12
            
            - | categorical:Optional[str] := Set the display of the
            categorical variable. (1) `None` ignores `Y` and only
            displays X pair-wise scatterplots, (2) 'hue' displays the
            categorical variable of `Y` as color and (3) 'size' displays
            the categorical variable with differently sized points
            
        Returns:
        =========
        
            - fig:matplotlib.pyplot.figure := The figure object, a FacetPlot 
            of scatterplots
            
       
       Raises:
       ========
        
            - WIP
            
            - ValueError 
    '''
    import math
    import itertools
    XY = pd.concat([Y, X], axis=1)
    idx_lvl= XY.columns.nlevels-1
    skip_multiindex = lambda df,i , idx_level=idx_lvl: df.columns[i][
        idx_level] if idx_level else df.columns[i]
    if idx_lvl:
        XY.columns = XY.columns.get_level_values(idx_lvl)
    x_combs = math.comb(X.shape[1], 2)
    MAX_ROWS = max_rows
    
    if categorical is not None:
        plots = x_combs*Y.shape[1]
    else:
        plots = x_combs
    ncols = cols
    size_scaling_factor = None
    total_rows = math.ceil(plots/ncols)
    total_figures = math.ceil(total_rows/MAX_ROWS)
    X_pairs = itertools.combinations(range(X.shape[1]), 2)
    subplots = itertools.product(X_pairs,range(Y.shape[1]) ) if \
        categorical is not None else X_pairs
    exhaustion_sentinel = False
    while True:
        if exhaustion_sentinel: break
        fig, axs = plt.subplots(nrows=MAX_ROWS, ncols=ncols, 
        figsize=figsize)
        ax_indices = itertools.product(range(MAX_ROWS), range(ncols), 
        repeat=1)
        plotslice = itertools.islice(subplots, MAX_ROWS*ncols)
        fig_generator = itertools.zip_longest( ax_indices, plotslice, 
        fillvalue = ((None, None), None) )
        for (axi, axj), e in fig_generator:
            e = tuple(flatten(e))
            if e[0] is not None:
                if categorical == 'size':
                    sns.scatterplot(x=skip_multiindex(X, e[0]), 
                    y=skip_multiindex(X, e[1]), data=XY,
                       size=skip_multiindex(Y, e[2]), ax=axs[axi, axj], 
                       legend=True)
                elif categorical == 'hue':
                    sns.scatterplot(x=skip_multiindex(X, e[0]), 
                    y=skip_multiindex(X, e[1]), data=XY,
                       hue=skip_multiindex(Y, e[2]), ax=axs[axi, axj], 
                       legend=True)
                elif categorical is None:
                    sns.scatterplot(x=skip_multiindex(X, e[0]), 
                    y=skip_multiindex(X, e[1]), data=XY,
                       ax=axs[axi, axj], legend=True)
                axs[axi, axj].set_xlabel(skip_multiindex(X, e[0]))
                axs[axi, axj].set_ylabel(skip_multiindex(X, e[1]))
                axs[axi, axj].xaxis.label.set_size(xaxis_label_size)
                axs[axi, axj].yaxis.label.set_size(yaxis_label_size)
                
            else:
                axs[axi, axj].axis('off')
                exhaustion_sentinel = True
        yield fig 
    return None



def corrspace_graph(df:pd.DataFrame):
    '''
        Generate a correlation graph representation of `df` datasets.
        Every node in the resulting graph is a variable and every 
        vertex between two nodes represents the correlation between
        the two variables, the correlation being the verted weight
        
        Args:
        ----
        
            - df:pandas.DataFrame := A dataset to depict
            
        Returns:
            - WIP
    '''
    
    raise NotImplemented()


def boxplots(df:pd.DataFrame, max_features:int=20, title:str="Boxplot",
             **kwargs)->typing.Generator[plt.Figure, None, None]:
    '''
        Wrapper around `pandas.boxplot` for datasets with a large number
        of features. Returns a generator of boxplots of subsets of the total
        features.
        
        Args:
        -----
        
            - df:pandas.DataFrame := The DataFrame to plot
            
            - max_features:int>0 := The maximum number of features to plot
            in each generated boxplot
            
            - title:str='Boxplot' := The title of the plot. A subplot index
            of the form **(X/Y)**, tracking the current plot will be
            appended prior to rendering
            
            - **kwargs:dict[str:Any] := Keyword arguments to be be forwarded
            to pandas.DataFrame.boxplot
            
        Returns:
        -------
        
            - Generator[plt.Figure, None, None] := A generator yielding 
            boxplots of feature subsets
    '''
    import math
    nplots = math.ceil(df.shape[1]/max_features)
    for i in range(nplots):
        if (i+1)*max_features<df.shape[1]:
            ndf = df.iloc[:,i*max_features:(i+1)*max_features]
        else:
            ndf = df.iloc[:,i*max_features:df.shape[1]]
        fig = ndf.boxplot(**kwargs)
        fig.set_title(title + " ({current}/{total})".format(current=i+1, 
        total=nplots) )
        yield fig

def advi_inspect_ELBO(fit_data, show:bool=True):
    '''
        Utility method to investigate ADVI fit.
        Returns log-ELBO against iterations.
    '''
    _h_advi_hist = fit_data.hist
    advi_elbo = pd.DataFrame(
        {'$log-ELBO$': -np.log(_h_advi_hist),
         'n': np.arange(_h_advi_hist.shape[0])})
    fig = sns.lineplot(y='$log-ELBO$', x='n', data=advi_elbo)
    if show:
        plt.show()
    return fig
