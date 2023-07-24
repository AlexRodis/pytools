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
import pytensor
import numpy as np 
import pytensor.tensor as at # For backwards compatibility reasons
from pymc.gp.cov import Covariance
import typing


class MultiLayerPerceptronKernel(Covariance):
    '''
        Multi layer Perceptron Kernel a.k.a Arcsine Kernel in
        :code:`pymc` and the :code:`pytensor` backend.

        A single layer perceptron is a Gaussian Process, whose kernel(s)
        were derived by Williams in `Computing with infinite neural
        networks
        <https://proceedings.neurips.cc/paper_files/paper/1996/file/ae5e3ce40e0404a45ecacaaf05e5f735-Paper.pdf>`_

        The commonly called "Multi Layer Perceptron Kernel" is a
        simplified implementation of this kernel. The resulting GP is
        equivalent to a single layer perceptron with the standard
        logistic function as its response function.

        .. math::

          k(x,y) = \\sigma^{2}\\frac{2}{\\pi }  \\text{asin} \\left (
          \\frac{ \\sigma_w^2 x^\\top y+\\sigma_b^2}{\\sqrt{
            \\sigma_w^2x^\\top x +\\sigma_b^2 + 1}\\sqrt{\\sigma_w^2
            y^\\top y + \\sigma_b^2 +1}} \\right )
          
        .. note::
        
            Implementation is a direct port of the GPy implementation of
            the same kernel. Automatic Relevance Determination (ARD) not
            implemented

        .. math::
        
        Attributes:
        ------------

            - | input_dims:int=1 := The number of input dimensions
                (columns) used for Covariance computations. Defaults to
                1 (use the first column only).

            - | active_dims:Optional[list[int]]=None := None or
                array-like specifying which input dimensions will be
                used in computing the covariance matrix. Can be
                specified as a boolean vector or a vector of indices.
                Optional. Defaults to None, and the first `input_dims`
                columns are selected.

            - | variance:Optional[float] := The output kernel variance
                scaling parameter, usually denoted `\eta`. Defaults to
                1.0. Must be positive.

            - | weight_variance:Optional[Union[float,
                np.typing.NDArray[float]]]=1.0. The variance of the
                weight in the A.N.N. Can be specified as either a scalar
                or a matrix of appropriate size (same as the first input
                matrix). Optional. Defaults to 1.0.

            - | bias_variance:Optional[Union[float,
                np.typing.NDArray[float]]]=1.0. The variance of the
                biases in the A.N.N. Can be either a scalar or a matrix
                of appropriate size. When computing covariance matrices
                (single input) must be a N-length vector of biases for a
                `N \times M` input matrix. When computing
                cross-covariance matrices, must be an `N \times N`
                matrix. Optional.  Defaults to 1.0.

            - | ARD:bool=False := A(utomatic)R(elevance)D(etermination)
                flag. Unused and raises an error if switched. Optional.
                Defaults to :code:`False`.

        Methods:
        -------

            Public:
            =======

                - | full(X:numpy.ndarray, Y:Optional[numpy.ndarray]) :=
                    Computes and returns the full covariance
                    matrix/kernel (Y=None) or the cross-covariance
                    matrix (Y=np.ndarray). The latter case is poorly
                    tested and may be wrong due to numericall stability 
                
                - | diag(X:typing.NDArray,Y:Optional[numpy.ndarray]) :=
                    Returns the diagonal of the covariance matrix
                    (Y=None) or the cross-covariance matrix
                    (Y=numpy.ndarray). The letter case is poorly tested
                    and may suffer from numerical stability issues.

    '''    
    four_over_tau:float = 2./np.pi
    
    
    def __init__(self,input_dims:int, active_dims:typing.Union[list[int], 
        int]=None, variance:float=1.0, weight_variance:typing.Union[float, 
        np.typing.NDArray]=1.0, bias_variance:typing.Union[float, 
        np.typing.NDArray]=1.0, ARD:bool=False)->None:
        super(MultiLayerPerceptronKernel, self).__init__(input_dims, 
        active_dims=active_dims)
        self.variance = variance
        self.weight_variance = weight_variance
        self.bias_variance = bias_variance
        if ARD:
            raise NotImplementedError(
                "Automatic Relevance Determination, not implemented")
        self.ARD = ARD
    
    def _comp_prod(self, X, X2=None):
        if X2 is None:
            return (at.math.square(X[:,self.active_dims])*self.weight_variance
            ).sum(axis=1)+self.bias_variance
        else:
            return (X[:,self.active_dims]*self.weight_variance).dot(X2[:,self.active_dims].T)+self.bias_variance
    
    def diag(self, X):
        """Compute the diagonal of the covariance matrix for X."""
        X_prod = self._comp_prod(X[:,self.active_dims])
        return self.variance*MultiLayerPerceptronKernel.four_over_tau*\
            at.math.arcsin(X_prod/(X_prod+1.))
    
    def full(self, X, X2=None):
        if X2 is None:
            X_denom = at.math.sqrt(self._comp_prod(X[:,self.active_dims])+1.)
            X2_denom = X_denom
            X2 = X
        else:
            X_denom = at.math.sqrt(self._comp_prod(X[:,self.active_dims])+1.)
            X2_denom = at.math.sqrt(self._comp_prod(X2)+1.)
        XTX = self._comp_prod(X[:,self.active_dims],X2)/X_denom[:,None]/X2_denom[None,:]
        return self.variance*MultiLayerPerceptronKernel.four_over_tau*\
            at.math.arcsin(XTX)
