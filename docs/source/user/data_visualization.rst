.. _chap-data-visualization:

Data Visualization
==================

Visualizing the data
--------------------

As for the HyperSpy data, the data objects :class:`~.signals.Stem2D` and :class:`~.signals.Stem3D` include a :meth:`~.signals.AbstractStemAcquisition.plot` method to display the data.

In addition to this basic function, the :class:`~.signals.Stem3D` class implements four other functions:

* :meth:`~.signals.Stem3D.plot_sum` which displays the sum of the 3D data along the last axis,
* :meth:`~.signals.Stem3D.plot_as2D` which displays the data as 2D data (the navigation direction is the "channel" axis while the data are spatial),
* :meth:`~.signals.Stem3D.plot_as1D` which displays the data as 1D data (the navigation directions are the spatial axes while the data is a spectrum, this is the behavior of the default :meth:`~.signals.AbstractStemAcquisition.plot` function),
* :meth:`~.signals.Stem3D.plot_roi` which considers the data as 1D and enable the user to mean the data over a spatial region-of-interest.

Visualizing the scan sampling mask
----------------------------------

As explained in :ref:`sec-basic-object-scan`, a Scan object can be represented with its sampling mask :math:`\mathbf{M}` defined as

    .. math::
        \mathbf{M}_i = 
        \begin{cases}
        1, & \text{if pixel \# $i$ is acquired}\\
        0, & \text{otherwise}\\
        \end{cases}

This representation suffer from information deficiency, but is interesting to study the acquired pixels repartition. This sampling mask which shape is the same as the spatial shape can be obtained using the :meth:`~.signals.Scan.get_mask` method of :class:`~.signals.Scan`. This one can also be plotted using the method :meth:`~.signals.Scan.plot`.