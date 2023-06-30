import unittest
import numpy
from src.pytools.math.kernels import MultiLayerPerceptronKernel
import GPy
import numpy as np
from itertools import product
from collections import namedtuple
from typing import NamedTuple

class TestKernels(unittest.TestCase):


    def test_MLPKernel(self):
        '''
            Test numerical stability of the MLP kernel. 
            Since implementation is based on GPy we assert
            :code:`GPy.kern.MLP` as ground truth
        '''
        output_variance:list[float] = [1.0, 5.0,3.5,10]
        weight_variance:list[float] = [.2,3.0,60,10,5.6]
        bias_variance:list[float] = [.5,1,5.1,20,12.3,]
        dims: list[int] = [1,5,3]
        X = numpy.random.rand(30,max(dims))
        params:tuple[float] = product(
                output_variance, weight_variance, 
                bias_variance, dims
                )
        TestParams = namedtuple('TestParams',[
            'output_variance',
            'weight_variance',
            'bias_variance',
            'dims'
            ])
        
        paramap = map(lambda e:TestParams(
                output_variance = e[0],
                weight_variance = e[1],
                bias_variance = e[2],
                dims = e[3]
            ), params)
        conditions:list[bool] = []
        for pars in  paramap:
            ref_obj = GPy.kern.MLP(
                    pars.dims,
                    pars.output_variance,
                    pars.weight_variance,
                    pars.bias_variance
                    )
            obj =  MultiLayerPerceptronKernel(
                    input_dims = pars.dims,
                    variance = pars.output_variance,
                    weight_variance = pars.weight_variance,
                    bias_variance = pars.bias_variance,
                    )
            full = obj.full(X).eval()
            diag = obj.diag(X).eval()
            cond:bool = np.allclose(full,ref_obj.K(X) )
            conditions.append(cond
                    )
        self.assertTrue(all(conditions))
