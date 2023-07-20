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

from pytools.typing import NDArray, FloatArray
import typing
from typing import Optional, Callable, Iterable, Sequence, Union, Any
from typing import Hashable, Iterator
from pytools.typing import Collection, CollectionClasses
import numpy as np
import pandas as pd
from dataclasses import dataclass
import cloudpickle
from warnings import warn
import collections
import math
import operator
import random
import itertools

@dataclass(slots=True)
class LinearSpace:
    r'''
        Data object representing a linear space. A linear space can be
        defined as the tuple (start, stop, n_points)
        
        Object Public Attributes:
        ==========================
        
            - | start:float := The start of the span. Inclusive
            
            - | stop:float := The end of the span. Inclusive
            
            - | n_points:Optional[float] := The number of evenly spaced
                points in the span. Optional. Defaults to 50
        
    '''
    start:float
    stop:float
    n_points:Optional[float] = 50
    
def matrix_meshgrid(*spans:Iterable[LinearSpace],
                    to_pandas:bool=False,
                    columns:Optional[Sequence[str]]=None
                    )->Union[FloatArray, pd.DataFrame]:
    r'''
        Wrapper around :code:`numpy.meshgrid` which returns a 2D matrix
        of the tidy format, as expected by most Machine Learning
        libraries
        
        Args:
        ======
        
            - | *spans:Iterable[LinearSequence] := An iterable of spans
                defining the coordinates of the meshgrid. Of the general
                form :code:`start`, :code:`stop`, :code:`[n_points]`
                
            - | to_pandas:bool=False := If :code:`True` convert the
                array into a :code:`pandas.DataFrame`. Optional and
                defaults to :code:`False` (returning an array).
                
            - | columns:Sequence[str] := Coordinate names for output
                dataframe columns. Optional. Ignored if
                :code:`to_pandas=False`.
            
        Returns:
        ========
        
            - | grid_matrix:NDArray[float] := The meshgrid in tidy
                matrix form
                
    '''
    coordinates:list[NDArray] = [
        np.linspace(
            span.start, span.stop, num=span.n_points
            ) for span in spans
    ]
    mesh = np.meshgrid(*coordinates)
    this = map(lambda elem: np.ravel(elem), mesh )
    grid_matrix = np.vstack(tuple(this)).T
    kargs:dict = dict(
        columns = columns
    ) if columns is not None else dict()
    return grid_matrix if not to_pandas else pd.DataFrame(
        grid_matrix, **kargs
    )
    

# Utility methods
def numpy_replace(arr:np.typing.NDArray, mapping:dict)->np.typing.NDArray:
    r'''
        Helper method providing pandas 'replace' functionality for
        numpy arrays.

        NOTE:
        ======

            Because numpy arrays are homogenous structures the
            replacement operation may change the structures datatype,
            most likely to the general `object`

        Args:
        =====

            - | arr:numpy.typing.NDArray := The array to operate on

            - | mapping:dict := A dictionary defining the replacement
               logic. Key are array values to replace, values define
               the replace with object
        
        Returns:
        ========

            - narr:np.typing.NDArray := Replaced array
    '''
    iarray=arr
    for i, name in mapping.items():
        iarray=np.where(iarray==i, name, iarray)
    return iarray


class Pipeline:
    r'''
        Implementation of :code:`sklearn.make_pipeline` without the final 
        estimator object

        Object Attributes:
        ==================

            - | *steps:tuple[Any] = Processing steps as objects. Must implement
                the `fit`, `transform` and `fit_transform` methods, per the
                standardized sklearn API
            
            - | _initialized:bool = False := Initialization flag, that ensures
                that the fit method has been called prior to tranformation

        Object Methods:
        ===============

            - | transform(array:numpy.typing.NDArray)->np.typing.NDArray := 
                Apply all preprocessing steps to `array` in the order they 
                were specified. The `fit` or `fit_transform` methods must 
                have been called first

            - | fit_transform(array:numpy.typing.NDArray) := Fit all estimators
                and transform the data. Sequencially calls the `fit` and 
                `tranform` methods from objects specified in `self.steps`
    '''
    def __init__(self, *steps):
        self.steps:tuple[typing.Callable] = steps
        self._initialized:bool=False
        
    def transform(self, arr):
        data = arr
        for step in self.steps:
            data = step.transform(data)
        return data

    def fit_transform(self, arr):
        data = arr
        for step in self.steps:
            data = step.fit_transform(data)
        return data


# Use the provided context manager instead
def render_df(df:pd.DataFrame):
    from IPython.display import display, HTML
    display(HTML(df.to_html()))
    return

def full_display(r=None, c=None):
    pd.set_option('display.max_rows', r)
    pd.set_option('display.max_columns',c)
    return

def reset_display():
    pd.set_option('display.max_rows',10)
    pd.set_option('display.max_columns',30)

def full_display_once(df:pd.DataFrame, r=None, c=None):
    full_display(r=r, c=c)
    render_df(df)
    reset_display()
