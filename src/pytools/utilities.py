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

from typing import NDArray
import numpy as np
import pandas as pd

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
