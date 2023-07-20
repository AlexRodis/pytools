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
def numpy_replace(arr:NDArray, mapping:dict)->NDArray:
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

            - narr:NDArray := Replaced array
    '''
    iarray=arr
    for i, name in mapping.items():
        iarray=np.where(iarray==i, name, iarray)
    return iarray


class Pipeline:
    r'''
        Implementation of :code:`sklearn.make_pipeline` without the
        final estimator object

        Object Attributes:
        ==================

            - | *steps:tuple[Any] = Processing steps as objects. Must
                implement the :code:`fit`, :code:`transform` and
                :code:`fit_transform` methods, per the standardized
                sklearn API
            
            - | _initialized:bool = False := Initialization flag, that
                ensures that the fit method has been called prior to
                transformation

        Object Methods:
        ===============

            - | transform(array:numpy.typing.NDArray)->NDArray := Apply
                all preprocessing steps to `array` in the order they
                were specified. The `fit` or `fit_transform` methods
                must have been called first

            - | fit_transform(array:NDArray) := Fit all
                estimators and transform the data. Sequentially calls
                the :code:`fit` and :code:`transform` methods from
                objects specified in :code:`self.steps`
    '''
    def __init__(self, *steps):
        self.steps:tuple[Callable] = steps
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


# NOTE: Use the provided context manager instead
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
    
def dict_merge(left:dict, right:dict, strict:bool=False)->dict:
    r'''
        Utility for joining dictionaries pre 3.10
        
        Args:
        =====
        
            - | left:dict[Hashable, Any] := The dictionary to merge to
            
            - | right:dict[Hashable, Any] := The dictionary to merge
                from
            
            - | strict:bool=False := If :code:`True` will raise is the
                dictionaries have overlapping keys. Optional and
                defaults to :code:`False`. For common keys, values of
                right dict will override those of the left dict
                
        Returns:
        ========
        
            - | merged_dict:dict[Hashable, Any] := Return a new dict
                with keys and values from both dicts. If the dicts have
                keys in common, those of the right dict will override
                those of the left dict (:code:`strict=False`)
                
        Raises:
        =======
        
            - | ValueError := If :code:`strict=True` and there are
                overlapping keys between the dicts
                
        Warns:
        ======
        
            - | UserWarning := If :code:`strict=True` and there are
                overlapping keys between both of the dicts
    '''
    intersection = set(left.keys()).intersection(right.keys())
    if intersection != set():
        if strict:
            raise ValueError((
                "Attempting to strictly merge dictionaries with \n"
                "common keys. Detected these common keys \n"
                f"{intersection}"
            ))
        else:
            warn(("Common keys found in both dictionaries. Keys \n"
                  "of the right will override those of the left"))
    mdict = {
        k:(
            right[k] if k in right.keys() else left[k]
            ) for k in set(left.keys()).union(set(right.keys()))
    }
    return mdict

def unfold_grid(grid:dict, mode:str="cartesian", ival:int=0
                )->list[dict]:
    r'''
        Utility method for grid search. 
        
        Specify the grid as a dictionary whose keys are variable names
        and whose values are values for the variable in the grid.
        Expands this grid into separate dict of key value pairs
        according to selected expansion strategy. For Cartesian
        expansion generates all possible combination of values for all
        variables. For Linear expansion generates points incrementing
        each variable, holding the others constant
        
        If a variable should remain at a fixed value, provide the value
        directly. Any dict value which is a collection of some sort will
        be treated as a grid parameter
        
        Example Usage:
        
        .. code-block:: Python 3
            from pytools.utilities import unfold_grid
            
            grid_spec = dict(
                var1 = [1,2,3],
                var2 = ["Hello", "World"],
                var_static = "Static Value",
            )
            cartesian_grid = unfold_grid(grid_spec)
            linear_grid = unfold_grid(grid_spec, mode='linear')
            linear_grid_second_value = unfold_grid(grid_spec, 
            mode ='linear', 
            ival = 1 # Keep the second value of all other variables 
            # during linear search
            )
            
            print(cartesian_grid)
            >> Output:
            >> [{'var_static': 'Static Value', 'var1': 1, 'var2': 'Hello'}, {'var_static': 'Static Value', 'var1': 1, 'var2': 'World'}, {'var_static': 'Static Value', 'var1': 2, 'var2': 'Hello'}, {'var_static': 'Static Value', 'var1': 2, 'var2': 'World'}, {'var_static': 'Static Value', 'var1': 3, 'var2': 'Hello'}, {'var_static': 'Static Value', 'var1': 3, 'var2': 'World'}]
            
            print(linear_grid)
            >> Output:
            >>[{'var1': 1, 'var_static': 'Static Value', 'var2': 'Hello'}, {'var1': 2, 'var_static': 'Static Value', 'var2': 'Hello'}, {'var1': 3, 'var_static': 'Static Value', 'var2': 'Hello'}, {'var2': 'Hello', 'var_static': 'Static Value', 'var1': 1}, {'var2': 'World', 'var_static': 'Static Value', 'var1': 1}]
            
            print(linear_grid_second_value)
            >> Output:
            >>[{'var1': 1, 'var_static': 'Static Value', 'var2': 'World'}, {'var1': 2, 'var_static': 'Static Value', 'var2': 'World'}, {'var1': 3, 'var_static': 'Static Value', 'var2': 'World'}, {'var2': 'Hello', 'var_static': 'Static Value', 'var1': 2}, {'var2': 'World', 'var_static': 'Static Value', 'var1': 2}]
            
        Args:
        =====
        
            - | grid:dict[Hashable, Any] := The dictionary specifying
                the grid to generate. Keys are variable names. Values
                are either a collection of some sort (when variable
                values are searched) or not, in which case they are
                treated at fixed
        
            - | mode:str='cartesian' := The strategy for grid expansion.
                Acceptable values are 'cartesian' or 'linear'. For
                'cartesian' will generate the full grid of all possible
                values of variables and combinations thereof. For
                mode='linear' will only increment each variable holding
                the others constant
            
            - | ival:int=0 := Index of the value for all variables that
                remain fixed. Only meaningful for :code:`mode='linear'`.
                Optional. Defaults to :code:`0` and the first element is
                used
                
        Returns:
        =========
        
            - Iterator[dict[Hashable,Any]] := An Iterator of
              dictionaries over each point in the grid.
              
        Raises:
        ========
                   
            - | ValueError := If :code:`mode` argument is anything other
                than :code:`'cartesian'` or :code:`'linear'`
              
            
            
    '''
    from itertools import product
    static = {
        k:v for k,v in grid.items() if not isinstance(
            v, CollectionClasses)
        }
    grided = {
        k:v for k,v in grid.items() if isinstance(
            v, CollectionClasses)
        }
    if mode == "linear":
        this = lambda exkey: {
            k:v[ival] for k,v in grided.items() if k!=exkey
            }
        return [{k:e}|static|this(k) for k,v in grided.items() for e in v]
    elif mode == "cartesian":
        grid_expansion = product(*map(lambda e: e[-1], grided.items()))
        expanded = map(
            lambda term: {
                k:v for k,v in zip(grided.keys(), term, strict=True)
                }, grid_expansion
            
            )
        return [static|elem for elem in expanded]
    else:
        raise ValueError((
            "Received unrecognized value for 'mode' argument. \n"
            f"One of 'cartesian' or 'linear' but received {mode}\n"
            "instead \n"
        ))    
    
    

# NOTE: Needs more extensive profiling to assess efficiency and possible
# alternate implementations
def add_row(df:pd.DataFrame, row:dict):
    r'''
        Convenience function for appending rows to and existing
        dataframe
        
        Example usage:
        
        .. code-block:: Python 3
            import numpy as np
            X = np.random.rand(101,5)
            x, X = X[-1,:], X[:-1, :]
            cols = [
                f"v{i}" for i in range(X.shape[-1])
            ]
            df = pd.DataFrame(X, columns=cols)
            row = {col:x[i] for i,col in enumerate(cols)}
            ndf = add_row(df, row)
            print(df.shape[0])
            >> Output
            >> 100
            print(ndf.shape[0])
            >> Output
            >> 101
        
        Args:
        =====
        
            - df:pandas.DataFrame := The dataframe to append to
            
            - | row:dict[str, Any] := The new row to append to the
                dataframe. Is dictionary of column names to values. Keys
                must exactly match the columns of df
        
        Returns:
        ========
        
            - ndf := The extended dataframe
    '''
    ndf = df.copy(deep=True)
    ndf.loc[len(ndf),:] = row
    return ndf

def save(obj, dir:str="obj.pickle"):
    r'''
        Utility method that uses :code:`cloudpickle` to serialize and
        save arbitrary Python objects.
        
        Args:
        =====
        
            - obj:Any := The object to serialize
            
            - | dir:str='obj.pickle' := The directory / file to save the
                object to
    '''
    with open(dir, 'wb') as file:
        cloudpickle.dump(obj, file)
        
def load(dir:str = 'obj.pickle'):
    r'''
        Utility method for loading arbitrary Python objects from disk
        
        Args:
        =====

            - | dir:str='obj.pickle' := Directory where the object is
              saved
              
        Returns:
        ========
        
            - | obj:Any := The un serialized object
    '''
    with open(dir, 'rb') as file:
        obj = cloudpickle.load(file)
    return obj


def value_counts(df:pd.DataFrame)->pd.DataFrame:
    r'''
        Return counts of unique values column-wise
        
        Returns an hierarchically indexed dataframe whose columns are
        column_labels x unique_values. Has a single row counting the
        number of times each unique value was seen
        
        Example usage:
        
        .. code-block :: Python 3
            import numpy as np
            import pandas as pd
            from pytools.utilities import value_counts
            
            X = np.asarray([
            [1, 0, 3, 3, 2],
            [0, 4, 2, 4, 4],
            [1, 3, 3, 1, 0],
            [4, 1, 2, 1, 2],
            [3, 4, 3, 3, 4],
            [4, 2, 3, 1, 4],
            [4, 1, 0, 3, 4],
                ])
            cols = [f"v{i}" for i in range(X.shape[-1])]
            df = pd.DataFrame(X,columns=cols)
            output = value_counts(df)
            print(output)
            >> Output
            >> v0          v1             v2       v3       v4      
            >>  0  1  3  4  0  1  2  3  4  0  2  3  1  3  4  0  2  4
            >> Value Counts  1  2  1  3  1  2  1  1  2  1  2  4  3  3  1  1  2  4
        
        Args:
        =====
        
            - df:pd.DataFrame := The DataFrame to operate on
        
        Returns:
        ========

            - | counts:pd.DataFrame := A new DataFrame, whose columns
                are counts of unique values of the original dataframe
    '''
    vcounts_df = pd.DataFrame(data = df.apply(lambda x: x.value_counts()).T.stack()).astype(int).T
    vcounts_df.index = ['Value Counts']
    return vcounts_df


def invert_dict(dictionary:dict)->dict:
    r'''
        Return a new dict with values and keys swapped. All values of
        the dict must be hashable
        
        Args:
        =====
        
            - | dictionary:dict[Hashable, Hashable] := The dictionary to
                invert
        
        Returns:
        ========
        
            - | ndict[Hashable, Hashable] := A new dictionary whose keys
                are the values of the input dict and whose values are
                the keys of the input dict

    '''
    return {v:k for k,v in dictionary.items()}

def jointly_discard_nan(*arrs, return_arrays:bool=False)->Union[tuple[pd.DataFrame],pd.Index ]:
    '''
       Return rows with non - missing data across all data frame
       
       Example Usage:
       
        .. code-block:: Python 3
            from numpy import np
            from pytools.utilities import jointly_discard_nan
            first = np.asarray([
                    [1,2,np.nan,1,3],
                    [1,2,1,1,2,],
                    [1,7,2,4,0],
                    [1,5,2,4,0],
            ])
            second = np.asarray([
                [1,2,0,np.nan,3],
                [1,2,1,1,2],
                [1,np.nan,2,4,0],
                [1,0,2,4,0],
                ])
            rows = jointly_discard_nan(
                first,
                second,
            )
            >> Output:
            >> Index([1,3], dtype=int64)

       Args:
       ======
       
           - | *arrs:pandas.DataFrame := One or more
               :code:`pandas.DataFrame`s to process
           
           - | return_arrays:bool=False := Selects whether to return
               full arrays or an indexer. Optional. Defaults to False
               (return only the indexer)
       
       Returns:
       =========
           
           - | nnan_idx:`pandas.Index` := A :code:`pandas.Index` object
               selecting the not_nan arrays
       
           - | narrs:tuple := A tuple of :code:`pandas.DataFrames`. Only
               if :code:`return_arrays=True`
       
       Warns:
       ======
           - If resulting index intersection is empty (no common, non-nan)
           elements between all arrays
           
       Raises:
       ------
       
           - | RuntimeError:= Generic sanity error raised when after the
               intersection any of the input arrays still have missing
               and nan elements
    '''
    from functools import reduce
    from warnings import warn
    not_nans=[arr.loc[~arr.isna().any(axis=1), :].index for arr in arrs]
    mask = reduce(lambda idx1, idx2: idx1.intersection(idx2), not_nans)
    if len(mask) == 0:
        raise warn('Index intersection is empty')
    if any(arr.loc[mask,:].isna().any(axis=1).any() for arr in arrs):
        raise RuntimeError(('Intersection failed! Resulting array intersection has'
                            ' missing elements! This is an error with the code'))
    if return_arrays:
        return tuple(arr.loc[mask,:] for arr in arrs)
    else:
        return mask