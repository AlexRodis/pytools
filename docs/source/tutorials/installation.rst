Installation
*************

The :code:`pytools` library can be installed from the public github
repo as

.. code-block:: Python3

    pip install git+https://github.com/AlexRodis/pytools

MCMC sampling via :code:`pymc` can be accelerated further by including
the optional dependencies :code:`numpyro` and :code:`jax`. Using the CPU
with :code:`jax` rather than the default implementation can be sometimes
be faster than the default :code:`pymc` one. Install with optional CPU
dependencies:

.. code-block:: Python3

    pip install -e "git+https://github.com/AlexRodis/pytools.git#egg=pytools[CPU]"

For CUDA capable environments :code:`jax` and CUDA can be used to further
accelerate MCMC. Install these dependencies as follows:

.. code-block:: Python3

    # CUDA 11
    pip install -e "git+https://github.com/AlexRodis/pytools.git#egg=pytools[GPU11]"
    
    # CUDA 12
    pip install -e "git+https://github.com/AlexRodis/pytools.git#egg=pytools[GPU12]"


And for TPU environments:

.. code-block:: Python3

    pip install -e "git+https://github.com/AlexRodis/pytools.git#egg=pytools[TPU]"

There is also the option to install with all optional dependencies for
development purposes

.. code-block:: Python3

    pip install -e "git+https://github.com/AlexRodis/pytools.git#egg=pytools[DEV]"