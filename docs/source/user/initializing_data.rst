Initializing the data
=====================

How to initialize the scan pattern
----------------------------------

The scan pattern can be initialized using three recipes:

* initialize it with the shape and path values (see :ref:`sec-basic-object-scan`),
* initialize it with a Numpy or HyperSpy file,
* initialize it as random sampling.
  
Recall that a path which is initialized with the data shape only is set to be a full raster (i.e. line-by-line) scan.

Recall also that all scan initialization functions allow to define a ratio argument (see :ref:`sec-basic-object-scan`). 

Initialize it with a file
~~~~~~~~~~~~~~~~~~~~~~~~~


The scan pattern can be initialized with a numpy :code:`.npz` file which should store:

* :code:`m` (resp. :code:`n`) which is the data number of rows (resp. columns), 
* :code:`path` which is the :code:`path` argument

To that end, one should use the :meth:`~.signals.Scan.from_file` method of :class:`~.signals.Scan`.

.. code-block:: python
    
    >>> import numpy as np
    >>> m, n = 50, 100
    >>> path = np.random.permutation(m*n)
    >>> data_2_save = {'m': m, 'n': n, 'path': path}
    >>> np.savez('my_scan.npz', **data_2_save)  # This saves the Scan numpy file

    >>> inpystem.Scan.from_file('my_scan.npz', ratio=0.5) # This loads the numpy scan file.
    <Scan, shape: (50, 100), ratio: 0.500>

Initialize it as random sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The sampling scan can last be initialized with the :meth:`~.signals.Scan.random` method of :class:`~.signals.Scan`. One should just give the spatial data shape :code:`(m, n)`. In addition to the ratio argument which can also be given, the user can give a seed to the method to have reproducible results.

.. code-block:: python

    >>> inpystem.Scan.random((50, 100))
    <Scan, shape: (50, 100), ratio: 1.000>
    >>> scan = inpystem.Scan.random((50, 100), ratio=0.2)
    >>> scan
    <Scan, shape: (50, 100), ratio: 0.200>
    >>> scan.path[:5]
    array([4071,  662, 4168, 3787, 4584])

    >>> scan = inpystem.Scan.random((50, 100), ratio=0.2, seed=0)
    >>> scan.path[:5]
    array([ 398, 3833, 4836, 4572,  636])
    >>> scan = inpystem.Scan.random((50, 100), ratio=0.2, seed=0)
    >>> scan.path[:5]  # This shows that setting the seed makes the results reproducible.
    array([ 398, 3833, 4836, 4572,  636])


Construct inpystem data manually
------------------------------

As explained in :ref:`sec-result-data`, the inpystem data is composed of a :class:`~.signals.Scan` object which defines the sampling pattern and the HyperSpy data which stores the data. Once both have been defined, the inpystem structure can be defined by hand.

.. code-block:: python

    >>> inpystem_data = inpystem.Stem2D(hsdata, scan=scan_object)


Construct inpystem data from a Numpy array
----------------------------------------

In case your image is a numpy array, one should define the HyperSpy data before creating the inpystem data.

.. code-block:: python

    >>> import numpy as np
    >>> import hyperspy.api as hs
    >>> shape = (50, 100, 1500)                 # This is the 3D data shape
    >>> im = np.ones(shape)                     # This is our image (which is 3D this time).
    >>> scan = inpystem.Scan.random(shape[:2])    # The scan is created (be careful to have 2-tuple shape).
    >>> hsdata = hs.signals.Signal1D(im)        # Here, hs data is created from numpy array.
    >>> inpystem.Stem3D(hsdata, scan)
    <Stem3D, title: , dimensions: (100, 50|1500), sampling ratio: 1.00>

Well, the problem here, which is the same as for numpy-based HyperSpy data, is that both :code:`axes_manager` and :code:`metadata` are empty. To correct that, it is hygly recommended to use a configuration file. That's the subject of next section.


Construct inpystem data from a configuration file
-----------------------------------------------

As explained in :ref:`sec-loading-data`, inpystem can load data from a :code:`.conf` configuration file. This is loaded by using the :func:`~.dataset.load_file` function (or the :func:`~.dataset.load_key` function if the configuration file is in the data path). To that end, a configuration file gives to inpystem all important informations.

First, the configuration file is separated in three main sections (case-sensitive, caution !):

* :code:`DATA 2D` for 2D data,
* :code:`DATA 3D` for 3D data,
* :code:`SCAN` for the scan pattern.

Among these sections, only one of :code:`DATA 2D` and :code:`DATA 3D` sections is required (if no data is given, inpystem can not do anything ...). And inside this section, the only key which is required is :code:`file` which specifies the location of the data file (numpy :code:`.npy` or .dm4 or all other file which is allowed by HyperSpy) **relative to the configuration file**. One info: contrary to sections wich are case-sensitive, keys are not.

In case no :code:`file` key is given inside a :code:`SCAN` section, the :func:`~.dataset.load_file` function **creates automatically a random scan object** (based on its :code:`scan_ratio` and :code:`scan_seed` arguments). Otherwise, a scan file (numpy or dm4/dm3) is loaded (the :code:`scan_ratio` argument of :func:`~.dataset.load_file` can still be given).

Hence, a basic configuration file could look like this.

.. code-block:: ini

    #
    # This is a demo file. 
    # This text is not used, that's a commentary.
    #
    
    [3D DATA]
    # This section defines all info about 3D data
    File = eels_data.dm4
    
    [SCAN] 
    # This section defines all info about scan pattern

    # If the following line is commented, the scan pattern would be random.
    FILE = scan.dm4

In the special case where the data file is a numpy :code:`.npy` file, one could define additional information to fill the HyperSpy :code:`axes_manager` attribute. To that end, a set of keys can be given inside the corresponding section. These keys should be like :code:`axis_dim_info` where:

* :code:`dim` is the axis index (0 for the :code:`x` axis, 1 for the :code:`y` axis and 2 in case of 3D data for the spectrum axis),
* :code:`info` belongs to :code:`name`, :code:`scale`, :code:`unit` and :code:`offset`.

As an example, the previous section data axes_manager should look like this.

.. code-block:: python

    >>> data = inpystem.Stem3D(hsdata, scan)
    Creating STEM acquisition...

    >>> data.hsdata.axes_manager
    <Axes manager, axes: (100, 50|1500)>
                Name |   size |  index |  offset |   scale |  units 
    ================ | ====== | ====== | ======= | ======= | ====== 
         <undefined> |    100 |      0 |       0 |       1 | <undefined> 
         <undefined> |     50 |      0 |       0 |       1 | <undefined> 
    ---------------- | ------ | ------ | ------- | ------- | ------ 
         <undefined> |   1500 |        |       0 |       1 | <undefined> 

If the numpy array is save inside a directory with the following configuration file, this issue would be fixed.

.. code-block:: ini

    #
    # This is a demo file to define Numpy data axes_manager. 
    #
    
    [3D DATA]
    file = numpy_data.npy

    # Infos for the axes_manager
    axis_0_name = x
    axis_1_name = y
    axis_2_name = Energy loss

    # Some more info for the energy loss axis
    axis_2_offset = 4.6e+02
    axis_2_scale = 0.32
    axis_2_unit = eV
    
    # No scan section, I want a random scan.

And the data would be loaded by simply typing this.

.. code-block:: python

    >>> inpystem.load_file('my-nice-file.conf', scan_ratio=0.5, scan_seed=0)


Loading example data for fast testing
-------------------------------------

Last way to load data, use one of the example data provided by inpystem. To that end, just use the :func:`~.dataset.load_example` function just as the :func:`~.dataset.load_key` with one of the following keys:

* :code:`'HR-sample'`: this is a real atomic-scale HAADF/EELS sample,
* :code:`'HR-synth'`: this is a synthetic EELS image generated to be similar to :code:`'HR-sample'`,
* :code:`'LR-synth'`: this is a synthetic low-resolution EELS image.

The first data were acquired in the context of the following works :cite:`a-zobelli2019spatial`, :cite:`a-preziosi2018direct`. Authors of these works would like to acknowledge Daniele Preziosi for the LAO-NNO thin film growth, Alexandre Gloter for the FIB lamella preparation and Xiaoyan Li for STEM experiments.

The two last data were generated to compare reconstruction methods in the context of STEM-EELS data inpainting :cite:`a-monier2018tci`. The high-resolution works were submitted.

References
----------

.. bibliography:: ../_static/references.bib
    :labelprefix: A
    :keyprefix: a-

