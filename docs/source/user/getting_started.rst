.. _chap-getting-started:

Getting started
===============

In this documentation, we will assume that the reader can write some command line or jupyter python.

Starting pystem in python
-------------------------

pystem can be imported in python just as any python package.

.. code-block:: python

    >>> import pystem

In addition to pystem, the `HyperSpy`_ library is required to construct the pystem objects as some arguments should be HyperSpy data.

.. code-block:: python

    >>> import hyperspy.api as hs

.. _HyperSpy: https://hyperspy.org/

Last, note that most of the scientific python applications require libraries such as numpy and matplotlib. It is recommended to import them as well.

.. code-block:: python

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

The base: What is a STEM acquisition here ?
-------------------------------------------

A STEM acquisition here is the result of two main informations:

* **The scan pattern** which is basically the array of the visited pixels indexes,
* **The data** which are the values of the data at the sampled pixels.

Be aware that the pystem library can handle 2D (e.g. HAADF data) as 3D data (e.g. EELS data). The data that are acquired at a spatial position (or pixel) can then be a single value (for 2D) or a spectrum (for 3D).


.. _sec-basic-object-scan:

The first basic object is the scan
----------------------------------

The basic object to understand is the :class:`~.signals.Scan` class. This is an object which stores the scan pattern. To create one, the data spatial shape and the path indexes should be given.

.. code-block:: python

    >>> shape = (3, 4)  # This is the spatial shape of the data : 3 rows, 4 columns.
    >>> path = [ 1,  3,  6,  4,  7, 11,  8,  0]  # These are the sampled pixels indexes.
    >>> scan = pystem.Scan(shape, path)
    >>> scan
    <Scan, shape: (3, 4), ratio: 0.667>

.. note:: Note that in this documentation, the matrix indexes are denoted in the rows major order (which is the order chosen in python). 

    It mean that if the spatial dimensions are :code:`(M, N)`, then the i'th index refers to the pixels which coordinates are :code:`(i // N, i % N)`. In the above example, the first visited pixel is the second pixel of the first line.

    Be careful also that **python coordinates begin at 0**, not 1.

Additionally, the :class:`~.signals.Scan` object allows you to select a fraction of the scan pattern. Let us consider a fully sampled acquisition which spatial shape is :code:`(M, N)`, then only 10% of the pixels can be selected by setting the :code:`ratio` argument to :code:`0.1`. This argument can be modified after the object definition.

Thought, be careful in case the acquisition is partially sampled (or example, if only :code:`r*M*N` pixels were acquired, with :code:`r` below 1). In such case, if :code:`ratio` is below :code:`r`, then only :code:`ratio*M*N` pixels are kept. If :code:`ratio` is above :code:`r`, then all :code:`r*M*N` pixels are kept.

.. code-block:: python

    >>> scan
    <Scan, shape: (3, 4), ratio: 0.667>
    >>> scan.ratio = 0.8  # Here, we ask for a ratio which is higher than 0.667.
    WARNING:root:Input ratio is higher than higher maximal ratio (0.667). Ratio is set to the maximal value.
    >>> scan.ratio = 0.5  # Here, the value is correct.
    >>> scan
    <Scan, shape: (3, 4), ratio: 0.500>

    >>> scan = pystem.Scan(shape, path, ratio=0.5)  # The ratio can be given at initialization.
    >>> scan
    <Scan, shape: (3, 4), ratio: 0.500>
    >>> scan.ratio = 0.667  # But don't worry, the additional visited pixels are not lost.
    >>> scan
    <Scan, shape: (3, 4), ratio: 0.667>

In fact, the pixels that are given at initialization of :code:`scan` are not lost when a below :code:`ratio` is given as the currently visited index are stored in :attr:`~.signals.Scan.path` attribute while the :attr:`~.signals.Scan.path_0` attribute stores all pixels at initialization.

.. code-block:: python

    >>> scan = pystem.Scan(shape, path, ratio=0.5)  # The ratio can be given at initialization.
    >>> scan
    <Scan, shape: (3, 4), ratio: 0.500>
    >>> scan.path
    .. code-block:: python
    >>> scan.path_0
    array([ 1,  3,  6,  4,  7, 11,  8,  0])
    >>> scan.ratio
    0.5

.. note:: 
    The Scan object data can be represented with a sampling mask :math:`\mathbf{M}` defined as

    .. math::
        \mathbf{M}_i = 
        \begin{cases}
        1, & \text{if pixel \# $i$ is acquired}\\
        0, & \text{otherwise}\\
        \end{cases}

    This representation suffer from information deficiency, but is interesting to study the acquired pixels repartition. This sampling mask which shape is the same as the spatial shape can be obtained using the :meth:`~.signals.Scan.get_mask` method of :class:`~.signals.Scan`. This one can also be plotted using the method :meth:`~.signals.Scan.plot` (see :ref:`chap-data-visualization`).




The second basic object is data
-------------------------------

Well, data here are nothing else than `HyperSpy`_ data. Please refer to its `documentation`_ for more info about it.

.. _documentation: http://http://hyperspy.org/hyperspy-doc/current/index.html

.. _sec-result-data:

The result is pystem data
-------------------------

As explained previously, the pystem data is the combination of a :class:`~.signals.Scan` object and an HyperSpy data. Two classes are proposed to the user:

* :class:`~.signals.Stem2D` for 2D data,
* :class:`~.signals.Stem2D` for 3D data.

Both are initialized with a scan pattern and the associated data. Though, the scan pattern is optional as the default scan pattern is raster scan (line-by-line) full sampling.

.. code-block:: python
    
     >>> import hyperspy.api as hs
     >>> haadf_hs = hs.load('haadf_data.dm4')
     >>> acquisition_1 = pystem.Stem2D(haadf_hs)  # fully sampled HAADF image.

     >>> m, n = haadf_hs.data.shape
     >>> N = int(0.5*m*n)  # The number of pixels to visit.
     >>> path = np.random.permutation(m*n)[:N]
     >>> scan = pystem.Scan((m, n), path)
     >>> acquisition_2 = pystem.Stem2D(haadf_hs, scan)  # partially sampled HAADF image.

     >>> eels_hs = hs.load('eels_data.dm4')
     >>> acquisition_3 = pystem.Stem3D(eels_hs)  # fully sampled EELS image.

.. _sec-loading-data:

Loading your data is faster
---------------------------

pystem offers you a way to accelerate the data definition. To that end, pystem proposes you to setup a data directory (let's say :code:`/my/wonderful/data/dir/`) and to put inside your data so that the structure looks like this:

::
 
     /my/wonderful/data/dir/
     |
     +-- MyData1
     |    |
     |    +-- ells_data.dm4
     |    +-- haadf_data.dm4
     |    +-- scan.dm4
     |    +-- MyData1.conf
     |
     +-- MyData2
          |
          +-- ells_data_2.dm4
          +-- MyData2.conf

.. note:: The data directory is not set by default. You should use the :func:`~.dataset.set_data_path` function to set the path. Then, it can be read with the :func:`~.dataset.read_data_path`.
    
    .. code-block:: python
    
        >>> pystem.set_data_path('/my/wonderful/data/dir/')
        >>> pystem.read_data_path()
        '/my/wonderful/data/dir/'



The data directory contains sub-directories which host:

* the data files (2D/3D data, scan pattern),
* the **configuration file** (such as ``MyData1.conf`` in the above tree).

The configuration file has the structure of a ``.ini`` file (have a look at `this page`_ for an example format) and defines the relative location of data files. This would look like this (be aware that the section names such as :code:`2D DATA` is case sensitive while keys such as :code:`file` are not).

.. code-block:: ini

    #
    # This is a demo MyData1.conf file
    #

    [2D DATA]
    # This section defines all info about 2D data
    file = haadf_data.dm4
    
    [3D DATA]
    # This section defines all info about 3D data
    File = eels_data.dm4
    
    [SCAN] 
    # This section defines all info about scan pattern
    FILE = scan.dm4

This file defines all is necessary to define the pystem data objects. To load the corresponding data, one should use the :func:`~.dataset.load_file` function which loads the data based on the :code:`.conf` configuration file. Alternatively, pystem can load the :code:`mydata.conf` data directly by using the :func:`~.dataset.load_key` with the :code:`mydata` key (as long as :code:`mydata.conf` is located inside the data directory). The difference between the two functions ? :func:`~.dataset.load_file` **allows you to load a file which is not in the data directory**.

In addition to the configuration file path, the user should specify which data to load with the :code:`ndim` argument (2 for 2D data and 3 for 3D data).

.. code-block:: python

    >>> pystem.get_data_path()
    /my/wonderful/data/dir/
    >>> acquisition = pystem.load_key('MyData1')
    >>> acquisition = pystem.load_file('/my/wonderful/data/dir/MyData2.conf', ndim=2)  

Other arguments (such as the scan pattern ratio) can be passed to the two load function. That will be seen later.


.. _this page: https://docs.python.org/3/library/configparser.html#supported-ini-file-structure

What about restoration ?
------------------------

Well, everything was loaded and is ready for reconstruction. Lets us consider that your acquisition was partially sampled with a ratio of 0.2. So, to use any reconstruction method, use the :meth:`~.signals.AbstractStemAcquisition.restore` method of pystem objects.

The methods to reconstruct the data include nearest neighbor interpolation, regularized least-square and dictionary learning. Let's try with an example data (pystem has three dataset that can be loaded easily, this will be mentioned in).

.. code-block:: python

    >>> import pystem
    >>> data = pystem.load_example('HR-sample', ndim=2, scan_ratio=0.2)  # This loads example data.
    Reading configuration file ...
    Generating data ...
    Creating STEM acquisition...
    Correcting STEM acquisition...

    >>> data
    <Stem2D, title: HR-sample, dimensions: (|113, 63), sampling ratio: 0.20>
    >>> reconstructed_data, info = data.restore('interpolation', parameters={'method': 'nearest'})
    >>> reconstructed_data
    <Signal2D, title: HR-sample, dimensions: (|113, 63)>  # 2D hs data.
    >>> info
    {'time': 0.012229681015014648}  # Execution time in sec.

Have a look at the reconstructed data which is an HyperSpy data. It means that the reconstructed data can analyzed with HyperSpy tools. Additional information are returned in the :code:`info` dictionary (for the nearest neighbor method, the only information that is returned is the execution time).

What about :code:`axes_manager` and :code:`metadata` informations ?
-------------------------------------------------------------------

The initialization of :class:`~.signals.Stem2D` or :class:`~.signals.Stem3D` objects need an HyperSpy image which stores information about the axes (as the :code:`axes_manager` attribute) and other general information (as the :code:`metadata` attribute). **These informations are transfered to the reconstructed data**.

.. code-block:: python

    >>> data.hsdata.metadata
    ├── General
    │   ├── original_filename = spim4-2-df-manualy aligned image.dm4
    │   └── title = HR-sample
    └── Signal
        ├── Noise_properties
        │   └── Variance_linear_model
        │       ├── gain_factor = 1.0
        │       └── gain_offset = 0.0
        ├── binned = False
        ├── quantity = Intensity
        └── signal_type = 
    >>> data.hsdata.axes_manager
    <Axes manager, axes: (|113, 63)>
                Name |   size |  index |  offset |   scale |  units 
    ================ | ====== | ====== | ======= | ======= | ====== 
    ---------------- | ------ | ------ | ------- | ------- | ------ 
                   x |    113 |        |      -0 |       1 |        
                   y |     63 |        |      -0 |       1 |        

    >>> reconstructed_data.metadata
    ├── General
    │   ├── original_filename = spim4-2-df-manualy aligned image.dm4
    │   └── title = HR-sample
    └── Signal
        ├── Noise_properties
        │   └── Variance_linear_model
        │       ├── gain_factor = 1.0
        │       └── gain_offset = 0.0
        ├── binned = False
        ├── quantity = Intensity
        └── signal_type = 
    >>> reconstructed_data.axes_manager
    <Axes manager, axes: (|113, 63)>
                Name |   size |  index |  offset |   scale |  units 
    ================ | ====== | ====== | ======= | ======= | ====== 
    ---------------- | ------ | ------ | ------- | ------- | ------ 
                   x |    113 |        |      -0 |       1 |        
                   y |     63 |        |      -0 |       1 |        


