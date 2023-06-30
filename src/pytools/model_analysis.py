#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# This module contains additional tools for Artificial Neural Network
# decision interpretation via masking. We can assess an ANNs decision
# making process by locating the minimum set of inputs that is necessary
# for the model to make correct predictions. Give a set of inputs that
# the ANN correctly predicts, we find the smallest possible set of inputs
# we can remove (setting to 0), such that predictions are still reasonably
# correct. Features can be masked by multiplying the validation matrix with
# a vector of 1s (feature persists) and 0s (feature is removed). We seek to
# find the vector with the smallest number of 1s. Utilities defined here
# all implement difference ways of generating candidate vectors:
#   - | CartesiaMasking := Generate all possible vectors of 1s and 0s.
#       becomes extremelly large as the number of features increases
#   - | CombinatorialMasking := All possible ways to mask K features.
#   - | SingleMasking := All possible ways to mask a single feature
#   -   at a time
#   - | RandomMasking := Generate a random subset of the full Cartesian
#   -   masking
#   - | RandomCombinatorialMasking := Generate a random subset of
#   -   of the full Combinatorial masking for K features 


import typing
import random
from abc import ABC, abstractmethod
from math import comb



# Utility generator classes
class RandomSampleGenerator:
    '''
        Generator implementation of `random.sample`. 

        See the docstring of `random.sample` for more 
        information

        Note:
        =====
        
        This is an infinite generator
        
        Attributes:
        ============


            - population:Sequence := The population to sample from
            
            - | counts:Sequence[int]:= population-length sequence of
                positive integers signaling possible repeats per population
                element. Optional
            
            - | k:int=1 := Possitive integer indicating the length of each
                sample
        
        Methods:
        =========
            - __next__ := yield a random sample of the population
            
    '''
    def __init__(self, population:typing.Sequence,
                 counts:typing.Optional[typing.Sequence[int]]=None, 
                 k:int=1):
        import random
        from functools import partial
        self._population=population
        self._counts=counts
        self._k=k
        self._sampler=partial(random.sample, self.population,
        counts=self.counts, k=self.k)
        
    @property
    def population(self):
        return self._population
    @population.setter
    def population(self,val:typing.Sequence[typing.Any]):
        self._population=val
    @property
    def counts(self):
        return self._counts
    @counts.setter
    def counts(self,val:typing.Sequence[int]):
        if all(val)>=1:
            raise ValueError(("Counts must be a Sequence of positive "
                                f"integers. Received {val} instead"))
        else:                      
            self._counts=val
    @property
    def sampler(self)->typing.Callable[None,tuple[int]]:
        return self._sampler
    @sampler.setter
    def sampler(self,val:typing.Any)->None:
        raise RuntimeError("Can't set attribute `sampler`")
        
    @property
    def k(self):
        return self._k
    @k.setter
    def k(self,val:int)->None:
        if val<1:
            raise ValueError("k must be a positive integer")
        else:
            self._k=val
            
    def __iter__(self):
        return iter(self)
    
    def __next__(self):
        return tuple(self.sampler())

class KRandomGenerator:
    '''
        Generate random combinations from the set
        
        .. math::
            (A^B|A,B)
            A={1,...}, B={0,...}, |A|=K,|B|=K-1
        
        Note:
        =====

        This is in infinite generator.
        
        Attributes:
        -----------
            
            - | random_seed:int:= Random seed as per
                code:`random.random_seed` for reproducible
                results
            
            - M:int:=Length of the output Sequence
            
            - | K:int=1:= Number of 1s in output Sequence.
                Must be a positive integer and K<=M. Optional.
                Defaults to 1
            
            - | sampler:Generator[tuple[int], None, None]:=
                The sample generator. This is in infinite
                Generator
        
        
        Methods:
        -------
        
            - __next__:= Yields the next sample
    '''
    
    def __init__(self,M:int, random_seed:typing.Optional[int]=None,
                K:int=1):
        import random
        from functools import partial
        if M<1:
            raise ValueError()
        if K<1:
            raise ValueError()
        self.random_seed=random_seed
        if random_seed is not None:
            random.random_seed(self.random_seed)
        self.M=M
        if K<1:
            raise ValueError("K must be greater than 0")
        elif K>M:
            raise ValueError(("K must be less or equal to M. "
                             f"Received {K}>{M} instead"))
        else:
            self.K=K
        self.sampler = partial(
            lambda k,m: tuple([0 if i not in [
                random.randint(0,m) for _ in range(0,k)] else 1 for i \
                    in range(0,61) ] ), self.K,self.M
        )        
    def __next__(self):
        return self.sampler()


# Input Masking Generators
class InputMasking:
    '''
        Base class for input masking strategies and
        implementations. Core attributes of any
        implementation are:
        
            - | shape:tuple[int, int]: = A tuple of ints
                describing the input feature dimensions
                (ignoring the `sample` dimension). For
                example (61,) for a 1D input of 61 input
                features or (255,255,3) for an image.
            
            - | core_generator:Generator[tuple[int],
                None, None] := Core generator expression
                for masking. The concrete subclass is
                responsible for its implementation and
                initialization. Must yield `shape`-length
                tuples of 1s and 0s
            
            - | max_iters:int:= Maximum amount of
                possible masks the generator can yield
            
            Other attributes are:
            
            - | typecaster:Any := A constructor the
                outputs will be cast to, i.e.`numpy.asarray`
            
            - | reverse:bool=False:= If `True` yields
                the masking in reverse order i.e. right to
                left. Otherwise defaults to `False` and masks
                elements in left-to-right. Only available for
                non-random masking strategies.
            
            - | iters:int=1 := Utility attribute counting
                the number of iterations
        
        WARNING!:
        =========

            
        Regardless of implementation details, for every
        strategies generator, element returned during the first
        iteration should be a fully unmasked vector of inputs to
        establish a baseline truth for future metrics. This is to be
        done even if it conflicts with the strategies' definition, i.e.
        n_unmasked<total features
            
    '''
    
    def __iter__(self):
        return self
    
    @abstractmethod
    def __next__(self):
        raise NotImplemented()
    
    @property
    @abstractmethod
    def shape(self):
        return self._shape
    @shape.setter
    @abstractmethod
    def shape(self, value):
        self._shape=value
    @property
    @abstractmethod
    def iters(self):
        return self._iters
    @iters.setter
    @abstractmethod
    def iters(self,val):
        self._iters=val
    @property
    @abstractmethod
    def max_iters(self):
        return self._max_iters
    @max_iters.setter
    @abstractmethod
    def max_iters(self, val):
        self._max_iters=val
    @property
    @abstractmethod
    def core_generator(self):
        return self._core_generator
    @core_generator.setter
    @abstractmethod
    def core_generator(self,val):
        raise AttributeError()
    @property
    @abstractmethod
    def reverse(self):
        return self._reverse
    @reverse.setter
    @abstractmethod
    def reverse(self,val:bool):
        self._reverse=val
    @property
    @abstractmethod
    def typecaster(self):
        return self._typecaster
    @typecaster.setter
    @abstractmethod
    def typecaster(self, typecaster):
        self._typecaster
    
    @staticmethod
    def one_of_K(K:int, reverse=False)->list[int]:
        '''
            Single input feature dropout generator.
            Results in masking vectors of the form
            (1,1,1,1,1,1,1....), (0,1,1,1,1,...)

            Args:
            -----

                - | K:int := The number of features to mask.
                    Currently only support rank-1 input feature
                    tensors

                - | reverse:bool=False := Whether to mask starting
                    at the rightmost element. Optional. Defaults to
                    False (mask left to right)


        '''
        ranger=lambda e:reversed(range(e)) if reverse else range(e)
        initial=True
        if initial:
            initial=False
            yield [1]*K
        for i in ranger(K):
            l=[1]*(K-1)
            l.insert(i,0)
            yield l
    

class CartesianMasking(InputMasking):
    '''
        Masking Generator that yields Cartesian Product
        masks,

        Generates all possible shape-length tuples of
        1's and 0's. Effectively yields every possible
        subset of input features aka a type of powerset. 
        Equivalent to a shape nested for loop
    '''
    
    MAX_PERMUTATIONS=lambda e:2**sum(e)
    
    
    def __init__(self, shape:tuple[int], typecaster=tuple,
        reverse:bool=False):
        import itertools
        import numpy as np
        if len(shape)>1:
            raise ValueError("Shape must be a length-1 tuple")
        else:
            self._shape=shape
        self._base_set=[0,1] if reverse else [1,0] 
        self._iters=1
        self._typecaster=typecaster
        self._max_iters=CartesianMasking.MAX_PERMUTATIONS(shape)
        self._reverse=reverse
        self._core_generator=itertools.product(self.base_set,
            repeat=np.asarray(self.shape).sum())
    
    @property
    def base_set(self):
        return self._base_set
    
    @base_set.setter
    def base_set(self,val):
        self._base_set=val
    
    def __next__(self):
        if self.iters>self.max_iters:
            raise StopIteration()
        else:
            self.iters+=1
            return self.typecaster(next(self.core_generator))


        
class SingleMasking(InputMasking):
    '''
        Single input knockout masking. 

        Yields all possible shape-1 subsets of input features 
        by progressively removing a single input feature. 
        Yields masking vectors of the form 
        :math`(1,1,1,1,...,1), (1,1,...,0), (1,1,...,0,1) ...`
    '''
    MAX_PERMUTATIONS=lambda e:sum(e)+1
            
    def __init__(self,shape:tuple[int], typecaster=tuple,
                reverse:bool=False):
        import numpy as np
        if len(shape)>1:
            raise ValueError("Shape must be a length-1 tuple")
        else:
            self._shape=shape
        self._iters=1
        self._typecaster=typecaster
        self._max_iters=SingleMasking.MAX_PERMUTATIONS(shape)
        self._reverse=reverse
        # Inherited from parent
        self._core_generator=one_of_K(sum(shape), reverse=self.reverse)
    
    def __next__(self):
        if self.iters==self.max_iters:
            raise StopIteration()
        else:
            self._iters+=1
            return self.typecaster(next(self.core_generator))

   
    
class RandomMasking(InputMasking):
    '''
        Random masking strategy implementation.

        The number and identity of features to
        be masked is randomized. Randomly sample
        the Cartesian Product of `A x B`, where
        A={1,...}, B={0,...}, |A|=|B|=`shape`
        
        Attributes:
        ----------
        
            - iters:int=1 := Counter the number of iterations
            
            - | typecaster:Callable[tuple[int],Any] := Constructor
                for output data structure. Optional. Defaults to `tuple`
            
            - | max_masks:Optional[int]=None := Sets the max_number of
                random masks that can be generated. Optional. Defaults to
                :code:`None` and the generator is infinite
            
            - | random_seed:Optional[int]:= If provided, sets the random
                seed for reproducible results. Optional. Defaults to `None`
            
            - | core_generator:Generator[tuple[int],None,None] := The core
                random_mask generator expression
            
        Methods:
        --------
        
            - __next__:= Yield the next random Cartesian Product sample
    '''
    def __init__(self,shape:tuple[int],
                 typecaster:typing.Callable[
                    tuple[int],typing.Any]=tuple,
                 max_masks:typing.Optional[int]=None,
                 random_seed:typing.Optional[int]=None):
        import random
        if len(shape)>1:
            raise ValueError("Shape must be a length-1 tuple")
        else:
            self._shape=shape
        self._iters=1
        self._typecaster=typecaster
        self._max_iters=max_masks
        self._random_seed=random_seed
        if self.random_seed is not None:
            random.seed(self.random_seed)
        self._core_generator= RandomSampleGenerator(
            [1,0], counts=[self.shape[0],self.shape[0]], 
            k=self.shape[0])
        
    @property
    def random_seed(self):
        return self._random_seed
    
    @random_seed.setter
    def random_seed(self,val:int)->None:
        self._random_seed=int(val)
        
    def __next__(self):
        if self.max_iters is None or self.iters<=self.max_iters:
            self.iters+=1
            return self.typecaster(next(self.core_generator))
        else:
            raise StopIteration()


class RandomCombinatorialMasking(InputMasking):
    '''
        Generate random masks of K input
        unmasked features. 

        This is effectively a random sampler from the set of
        combinations of the set:
        
        .. math::
        
            A^B, A={1,...},B={0,...}, |A|=K, |B|=shape-K
    '''
    
    def __init__(self,shape:tuple[int], typecaster=tuple,
                 n_unmasked:int=1,
                max_masks:typing.Optional[int]=None):
        if len(shape)>1:
            raise ValueError("Shape must be a length-1 tuple")
        else:
            self._shape=shape
        self._iters=1
        self._typecaster=typecaster
        self._max_iters=max_masks
        self._n_unmasked=n_unmasked
        self._core_generator = KRandomGenerator(
            self.shape[0],K=self.n_unmasked)
        
    @property
    def n_unmasked(self):
        return self._n_unmasked
    
    @n_unmasked.setter
    def n_unmasked(self,val:int):
        if val<1:
            raise ValueError(
                ("Number of unmasked elements must be positive integer")
            )
        else:
            self._n_unmasked=val
    
    def __next__(self):
        if self.iters==1:
            self.iters+=1
            return self.typecaster(
                tuple([1]*self.shape[0])
            )
        elif  self.max_iters is None or self.iters<=self.max_iters:
            self.iters+=1
            return self.typecaster(next(self.core_generator))
        else:
            raise StopIteration()
            
            
class CombinatorialMasking(InputMasking):
    
    '''
        Combinatorial masking of K features.
        
        Generates the complete set of possible
        masks of K input features.
        
        Attributes:
        -----------
        
            - iters:int=1 := Number of iterations
            
            - | typecaster:= Output data structure
                constructor
            
            - | n_unmasked:int=1 := Set the number of
                unmasked elements to generate. Optional.
                Defaults to 1 and is the generator is
                equivalent to `SingleMasking`.
            
            - | max_iters:int := The maximum number of
                masks that will be generated.
            
            - | reverse:bool=False:= Selects whether to
                mask left-to-right (False) or right-to-left
                (True). Optional. Defaults to `True`.
            
            - | masked_index_generateor:Generator[
                tuple[int],None,None]:=Generator the the
                indices of unmasked elements in the output
                vector
            
            - | core_generator:Generator[tuple[int],
                None,None]:= Generator yielding masking
                vectors
    '''
    
    MAX_PERMUTATIONS=lambda n,k:comb(n,k)
    
    def __init__(self,shape:tuple[int], typecaster=tuple,
                 n_unmasked:int=1,reverse:bool=False):
        from itertools import combinations
        from math import comb
        if len(shape)>1:
            raise ValueError("Shape must be a length-1 tuple")
        else:
            self._shape=shape
        self._iters=1
        self._typecaster=typecaster
        self._n_unmasked=n_unmasked
        self._reverse=reverse
        self._max_iters=comb(self.shape[0], self.n_unmasked)
        self._masked_index_generator=combinations(
            reversed(range(self.shape[0])),r=self.n_unmasked) if \
                self.reverse else combinations(range(self.shape[0]),
                r=self.n_unmasked)
        self._core_generator = (tuple([0 if i not in indices else 1 for\
             i in range(self.shape[0])]) \
                for indices in self._masked_index_generator)
    
    @property
    def n_unmasked(self)->int:
        return self._n_unmasked
    @n_unmasked.setter
    def n_unmasked(self,num:int)->None:
        if num<1:
            raise ValueError(("`n_unmasked` must be a positive integer."
                              f" Received {num} instead"))
        else:
            self._n_unmasked=num
            
    def __next__(self):
        if self.iters>self.max_iters+1:
            raise StopIteration()

        if self.iters==1:
            self.iters+=1
            return self.typecaster(
                tuple([1]*self.shape[0])
            )
        else:
            self.iters+=1
            return self.typecaster(next(self.core_generator))
