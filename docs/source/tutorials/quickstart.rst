Quickstart
***********

Installation
=============

.. include:: tutorials/installation.rst

Example usage

.. code-block:: Python3
    
    from pytools.utilities import matrix_meshgrid, LinearSpace

    # Create a simple meshgrid in as a 2D matrix
    mesh = matrix_meshgrid(
            LinearSpace(0,1, 50),
            LinearSpace(0,10,100),
            LinearSpace(0,10)
    )
    print(mesh)
    # Output:
    # [[ 0.          0.          0.        ]
    # [ 0.          0.          0.20408163]
    # [ 0.          0.          0.40816327]
    # ...
    # [ 1.         10.          9.59183673]
    # [ 1.         10.          9.79591837]
    # [ 1.         10.         10.        ]]
 


