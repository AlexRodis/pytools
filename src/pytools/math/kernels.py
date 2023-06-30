import pytensor
import numpy as np 
import pytensor.tensor as at # For backwards compatibility reasons
from pymc.gp.cov import Covariance
import typing

# Feature Implementation Note: The original code implements Automatic
# Relevance Determination (ARD). It would be interesting to see if this
# can be implemented in a Bayesian framework

# Maintenance Note: This was written the `aesara` as the `pymc` backend.
# Need to update it to `pytensor` backend

class MultiLayerPerceptronKernel(Covariance):
    '''
        Multi layer Perceptron Kernel a.k.a Arcsine Kernel in 
        :code:`pymc` and the :code:`pytensor` backend.

        A single layer perceptor is a Gaussian Process, whose
        kernel(s) were derived by Williams in [INSERT REF HERE]i

        The commonly called "Multi Layer Perceptron Kernel" is a
        simplified implementation of this kernel. The resulting
        GP is equivalent to a single layer perceptron with the
        standard logistic function as its response function.

        .. math::

          k(x,y) = \\sigma^{2}\\frac{2}{\\pi }  \\text{asin} \\left
          ( \\frac{ \\sigma_w^2 x^\\top y+\\sigma_b^2}{\\sqrt{
            \\sigma_w^2x^\\top x +\\sigma_b^2 + 1}\\sqrt{\\sigma_w^2 
            y^\\top y + \\sigma_b^2 +1}} \\right )
          
        NOTE:
        =====
        Implementation is a direct port of the GPy implementation
        of the same kernel. Automatic Relevance Determination (ARD)
        not implemented

        .. math::
        
        Attributes:
        ------------

            - | input_dims:int=1 := The number of input dimensions (columns)
                used for Covariance computations. Defaults to 1 (use the first 
                column only).

            - | active_dims:Optional[list[int]]=None := None or array-like 
                specifying which input dimensions will be used in computing the 
                covariance matrix. Can be specified as a boolean vector or a 
                vector of indices. Optional. Defaults to None, and the first
                `input_dims` columns are selected.

            - | variance:Optional[float] := The output kernel variance scaling 
                parameter, usually denoted `\eta`. Defaults to 1.0. Must be 
                positive.

            - | weight_variance:Optional[Union[float, 
                np.typing.NDArray[float]]]=1.0. The variance of the weight in the 
                A.N.N. Can be specified as either a scalar or a matrix of 
                appropriate size (same as the first input matrix). Optional.  
                Defaults to 1.0.

            - | bias_variance:Optional[Union[float, 
                np.typing.NDArray[float]]]=1.0. The variance of the biases in the 
                A.N.N. Can be either a scalar or a matrix of appropriate size. 
                When computing covariance matrices (single input) must be a 
                N-length vector of biases for a `N \times M` input matrix. When 
                computing cross-covariance matrices, must be an `N \times N` 
                matrix. Optional.  Defaults to 1.0.

            - | ARD:bool=False := A(utomatic)R(elevance)D(etermination) flag. 
                Unused and raises an error if switched. Optional. Defaults to 
                :code:`False`.

        Methods:
        -------

            Public:
            =======

                - | full(X:numpy.ndarray, Y:Optional[numpy.ndarray]) := Computes
                    and returns the full covariance matrix/kernel (Y=None) or the
                    cross-covariance matrix (Y=np.ndarray). The latter case is
                    poorly tested and may be wrong due to numericall stability 
                
                - | diag(X:typing.NDArray,Y:Optional[numpy.ndarray]) := Returns 
                    the diagonal of the covariance matrix (Y=None) or the 
                    cross-covariance matrix (Y=numpy.ndarray). The letter case is 
                    poorly tested and may suffer from numerical stability issues.

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
            return (at.math.square(X)*self.weight_variance
            ).sum(axis=1)+self.bias_variance
        else:
            return (X*self.weight_variance).dot(X2.T)+self.bias_variance
    
    def diag(self, X):
        """Compute the diagonal of the covariance matrix for X."""
        X_prod = self._comp_prod(X)
        return self.variance*MultiLayerPerceptronKernel.four_over_tau*\
            at.math.arcsin(X_prod/(X_prod+1.))
    
    def full(self, X, X2=None):
        if X2 is None:
            X_denom = at.math.sqrt(self._comp_prod(X)+1.)
            X2_denom = X_denom
            X2 = X
        else:
            X_denom = at.math.sqrt(self._comp_prod(X)+1.)
            X2_denom = at.math.sqrt(self._comp_prod(X2)+1.)
        XTX = self._comp_prod(X,X2)/X_denom[:,None]/X2_denom[None,:]
        return self.variance*MultiLayerPerceptronKernel.four_over_tau*\
            at.math.arcsin(XTX)
