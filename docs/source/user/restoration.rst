Restoration
===========

Welcome to the main page of this documentation. The inpystem library is nothing else than a pluggin to HyperSpy to allow reconstruction.

Some words about pre and post-processing steps
----------------------------------------------

For 2D data
~~~~~~~~~~~

2D data are centered and normalized before reconstruction. The reason is to avoid highly variable reconstruction methods parameters.

.. code-block:: python

    # data numpy array is ready to be reconstructed.
    #

    # Let's get data mean and standard devition. 
    data_mean, data_std = data.mean(), data.std()

    # Let's center and normalize the data
    data_ready = (data - data_mean) / data_std

    #
    # Reconstruction is performed
    #

    # Let's perform the inverse transformation
    data_out = reconstructed_data * data_std + data_mean

    # data_out is returned

For 3D data
~~~~~~~~~~~

3D data have the save pre and post-processing as for 2D data. But in addition to centering and normalization, a **thresholded principal component analysis** (PCA) is performed to reduce the data dimension along the bands axis and to ensure the low rank assumption. This one states that most multi-band data result in the mixing of :code:`R` basic data with :code:`R` small compared to the data dimension).

The default behavior is to perform PCA with an automatically estimated threshold (which can be really over-estimated in case of data starvation situations, i.e. if you have almost as many samples as the data size). Though, the user can set both parameters and choose if PCA should be performed and which value the threshold should have.

To sum up, the steps are:

* perform thresholded PCA if required,
* center and normalize the data,
* perform reconstruction,
* re-set the data mean and standard deviation values,
* perform inverse PCA.


How to reconstruct my data
--------------------------

To reconstruct the data, the user should use the :meth:`~.signals.AbstractStem.restore` method. Both :class:`~.signals.Stem2D` and :class:`~.signals.Stem3D` classes need the following arguments to restore their data:

* :code:`method` which is the method name (default is :code:`'interpolation'`),
* :code:`parameters` which should be a dictionary with input parameters,
* :code:`verbose` which allows the method to display information on the console (default is :code:`True`).

In addition to these common parameters, the :class:`~.signals.Stem3D` class has the foolowing inputs:

* :code:`PCA_transform` which controlls the PCA execution (defualt is True for PCA execution),
* :code:`PCA_th` which states the PCA threshold.

A common reconstruction task will then look like this.

.. code-block:: python

    >>> data = inpystem.load_example('HR-sample', ndim=2, scan_ratio=0.2)
    Reading configuration file ...
    Generating data ...
    Creating STEM acquisition...
    Correcting STEM acquisition...

    >>> rec, info = data.restore()
    Restoring the 2D STEM acquisition...
    -- Interpolation reconstruction algorithm --
    Done in 0.01s.
    ---
    >>> rec
    <Signal2D, title: HR-sample, dimensions: (|113, 63)>
    >>> info
    {'time': 0.011758089065551758}


The reconstruction methods available
------------------------------------

All you need to know for each method is:

* what the method do (of course you need to know a little about it),
* his nickname to give to :meth:`~.signals.AbstractStem.restore`,
* his parameters,
* what informations are returned.



Restoration cheet sheet
~~~~~~~~~~~~~~~~~~~~~~~

+-----------------------+----+----+-----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
| Method input          | 2D | 3D | Parameters                                                                                                      | Output info                                                                      |
+=======================+====+====+=================================================================================================================+==================================================================================+
| :code:`interpolation` | x  | x  | :code:`method` (among :code:`nearest`, :code:`linear` and :code:`cubic`)                                        | :code:`time`                                                                     |
+-----------------------+----+----+-----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
| :code:`L1`            | x  |    | :code:`Lambda`, :code:`Nit`, :code:`init`                                                                       | :code:`E`, :code:`Gamma`, :code:`nnz_ratio`, :code:`time`                        |
+-----------------------+----+----+-----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
| :code:`3S`            |    | x  | :code:`Lambda`, :code:`scale`, :code:`Nit`, :code:`init`                                                        | :code:`E`, :code:`time`                                                          |
+-----------------------+----+----+-----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
| :code:`SNN`           |    | x  | :code:`Lambda`, :code:`Mu`, :code:`Nit`, :code:`init`                                                           | :code:`E`, :code:`time`                                                          |
+-----------------------+----+----+-----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
| :code:`CLS`           |    | x  | :code:`Lambda`, :code:`Nit`, :code:`init`                                                                       | :code:`E`, :code:`Gamma`, :code:`nnz_ratio`, :code:`time`                        |
+-----------------------+----+----+-----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
| :code:`Post_LS_CLS`   |    | x  | :code:`Lambda`, :code:`Nit`, :code:`init`                                                                       | :code:`E_CLS`, :code:`E_post_ls`, :code:`Gamma`, :code:`nnz_ratio`, :code:`time` |
+-----------------------+----+----+-----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
| :code:`ITKrMM`        | x  | x  | :code:`PatchSize`, :code:`K`, :code:`L`, :code:`S`, :code:`Nit_lr`, :code:`Nit`, :code:`init`, :code:`CLS_init` | :code:`dico`, :code:`E`, :code:`time`                                            |
+-----------------------+----+----+-----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
| :code:`ITKrMM_matlab` | x  | x  | :code:`PatchSize`, :code:`K`, :code:`L`, :code:`S`, :code:`Nit_lr`, :code:`Nit`, :code:`init`, :code:`CLS_init` | :code:`dico`, :code:`E`, :code:`time`                                            |
+-----------------------+----+----+-----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
| :code:`wKSVD`         | x  | x  | :code:`PatchSize`, :code:`K`, :code:`L`, :code:`S`, :code:`Nit_lr`, :code:`Nit`, :code:`init`, :code:`CLS_init` | :code:`dico`, :code:`E`, :code:`time`                                            |
+-----------------------+----+----+-----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
| :code:`wKSVD_matlab`  | x  | x  | :code:`PatchSize`, :code:`K`, :code:`L`, :code:`S`, :code:`Nit_lr`, :code:`Nit`, :code:`init`, :code:`CLS_init` | :code:`dico`, :code:`E`, :code:`time`                                            |
+-----------------------+----+----+-----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
| :code:`BPFA_matlab`   | x  | x  | :code:`PatchSize`, :code:`K`, :code:`step`, :code:`Nit`                                                         | :code:`dico`, :code:`time`                                                       |
+-----------------------+----+----+-----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+


Additional info in case :code:`PCA_transform` is :code:`True` is :code:`PCA_info` which stores the following keys:

* :code:`H`: the truncated PCA basis,
* :code:`PCA_th`: the PCA threshold,
* :code:`Ym`: the data mean.


Interpolation
~~~~~~~~~~~~~

The interpolation method calls linear, cubic or nearest neighbor interpolation.

The method to give to the :meth:`~.signals.AbstractStem.restore` method is :code:`interpolation`. The associated function is resp. :func:`~.restore.interpolation.interpolate`.

The input parameters are:

* :code:`method`: (optional, str) The interpolation method (among :code:`nearest`, :code:`linear` and :code:`cubic`). Default is nearest neighbor.

The output dictionary stores the following informations:

* :code:`time`: the execution time (in sec.),
* :code:`PCA_info`: in case of 3D data with PCA pre-processing, it stores info about PCA.

L1
~~~

This regularized least-square method solves the following optimization problem:

.. math::

    \gdef \x {\mathbf{x}}
    \gdef \y {\mathbf{y}}
    \hat{\x} = \mathrm{arg}\min_{ \x\in\mathbb{R}^{m \times n} }
           \frac{1}{2} ||(\x-\y)\cdot \Phi||_F^2 +
           \lambda ||\x\Psi||_1

where :math:`\mathbf{y}` are the corrupted data,  :math:`\Phi` is a subsampling operator and :math:`\Psi` is a 2D DCT operator. 

The method to give to the :meth:`~.signals.AbstractStem.restore` method is :code:`L1`. The associated function is resp. :func:`~.restore.LS_2D.L1_LS`.

The input parameters are:

* :code:`Lambda`: (float) The regularization parameter,
* :code:`init`: (optional, numpy array) An initial point for the gradient descent algorithm which should have the same shape as the input data,
* :code:`Nit`: (optional, int) The number of iterations.

The output dictionary stores the following informations:

* :code:`E`: The evolution of the functional value,
* :code:`Gamma`: The set of all pixel positions which coefficient in the DCT basis is non-zero,
* :code:`nnz-ratio`: The ratio of non-zero coefficients over the number of DCT coefficients,
* :code:`time`: the execution time (in sec.).


Smoothed SubSpace
~~~~~~~~~~~~~~~~~

The 3S algorithm denoise or reconstructs a multi-band image possibly
spatially sub-sampled in the case of spatially smooth images. It is
well adapted to intermediate scale images.

This algorithm performs a PCA pre-processing operation to estimate:

* the data subspace basis :math:`\mathbf{H}`,
* the subspace dimension :math:`R`,
* the associated eigenvalues in decreasing order :math:`\mathbf{d}`,
* the noise level :math:`\hat{\sigma}`.

After this estimation step, the algorithm solves the folowing
regularization problem in the PCA space:

.. math::

    \gdef \S {\mathbf{S}}
    \gdef \Y {\mathbf{Y}}
    \gdef \H {\mathbf{H}}
    \gdef \I {\mathcal{I}}

    \begin{aligned}
    \hat{\S} &= \underset{\S\in\mathbb{R}^{m \times n \times R}}{\arg\min}
            \frac{1}{2R}\left\|\S \mathbf{D}\right\|_\mathrm{F}^2 +
            \frac{\lambda}{2}\sum_{m=1}^{R} w_{m} |\S_{m,:}|_2^2\\
     &\textrm{s.t.}\quad
            \frac{1}{R}|\H_{1:R}^T\Y_{\I(n)}-\S_{\mathcal{I}(n)}|^2_2
            \leq\alpha\hat{\sigma}^2,\ \forall n
            \in \{1, \dots,\ m*n\}
    \end{aligned}


where :math:`\mathbf{Y}` are the corrupted data,  :math:`\mathbf{D}`
is a spatial finite difference operator and :math:`\mathcal{I}` is
the set of all sampled pixels. The coefficient :math:`\alpha` is a coefficient which scales the power of the data fidelity term.

For more details, see :cite:`b-monier2018tci`.           

The method to give to the :meth:`~.signals.AbstractStem.restore` method is :code:`3S`. The associated function is resp. :func:`~.restore.LS_3D.SSS`.

The input parameters are:

* :code:`Lambda`: (float) The regularization parameter,
* :code:`scale`: (optional, float) The spectr 
* :code:`init`: (optional, numpy array) An initial point for the gradient descent algorithm which should have the same shape as the input data,
* :code:`Nit`: (optional, int) The number of iterations.

The output dictionary stores the following informations:

* :code:`E`: The evolution of the functional value,
* :code:`time`: the execution time (in sec.),
* :code:`PCA_info`: in case of 3D data with PCA pre-processing, it stores info about PCA.


Smoothed Nuclear Norm
~~~~~~~~~~~~~~~~~~~~~

The SNN algorithm denoise or reconstructs a multi-band image possibly
spatially sub-sampled in the case of spatially smooth images. It is
well adapted to intermediate scale images.

This algorithm solves the folowing optimization problem:

.. math::

    \gdef \X {\mathbf{X}}
    \gdef \Y {\mathbf{Y}}
    \gdef \H {\mathbf{H}}
    \gdef \I {\mathcal{I}}

    \hat{\X} = \underset{\X\in\mathbb{R}^{m \times n \times B}}{\arg\min}
        \frac{1}{2}||\Y_\I - \X_\I||_\mathrm{F}^2 +
        \frac{\lambda}{2}\left\|\X \mathbf{D}\right\|_\mathrm{F}^2 +
        \mu ||\X||_*

where :math:`\mathbf{Y}` are the corrupted data,  :math:`\mathbf{D}`
is a spatial finite difference operator and :math:`\mathcal{I}` is
the set of all sampled pixels.

For more details, see :cite:`b-monier2018tci`.           

The method to give to the :meth:`~.signals.AbstractStem.restore` method is :code:`SNN`. The associated function is resp. :func:`~.restore.LS_3D.SNN`.

The input parameters are:

* :code:`Lambda`: (float) The :math:`\lambda` regularization parameter,
* :code:`Mu`: (float) The :math:`\mu` regularization parameter,
* :code:`init`: (optional, numpy array) An initial point for the gradient descent algorithm which should have the same shape as the input data,
* :code:`Nit`: (optional, int) The number of iterations.

The output dictionary stores the following informations:

* :code:`E`: The evolution of the functional value,
* :code:`time`: the execution time (in sec.),
* :code:`PCA_info`: in case of 3D data with PCA pre-processing, it stores info about PCA.


Cosine Least Square
~~~~~~~~~~~~~~~~~~~

The CLS algorithm denoises or reconstructs a multi-band image possibly
spatially sub-sampled in the case of spatially sparse content in the DCT
basis. It is well adapted to periodic data.

This algorithm solves the folowing optimization problem:

.. math::

    \gdef \X {\mathbf{X}}
    \gdef \Y {\mathbf{Y}}
    \gdef \H {\mathbf{H}}
    \gdef \I {\mathcal{I}}

    \hat{\X} = \underset{\X\in\mathbb{R}^{m \times n \times B}}{\arg\min}
        \frac{1}{2}||\Y_\I - \X_\I||_\mathrm{F}^2 +
        \lambda ||\X \Psi||_{2, 1}


where :math:`\mathbf{Y}` are the corrupted data,  :math:`\mathbf{D}`
is a spatial finite difference operator and :math:`\mathcal{I}` is
the set of all sampled pixels.

The method to give to the :meth:`~.signals.AbstractStem.restore` method is :code:`CLS`. The associated function is resp. :func:`~.restore.LS_CLS.CLS`.

The input parameters are:

* :code:`Lambda`: (float) The :math:`\lambda` regularization parameter,
* :code:`init`: (optional, numpy array) An initial point for the gradient descent algorithm which should have the same shape as the input data,
* :code:`Nit`: (optional, int) The number of iterations.

The output dictionary stores the following informations:

* :code:`E`: The evolution of the functional value,
* :code:`Gamma`: The set of all pixel positions which coefficient in the DCT basis is non-zero,
* :code:`nnz-ratio`: The ratio of non-zero coefficients over the number of DCT coefficients,
* :code:`time`: the execution time (in sec.),
* :code:`PCA_info`: in case of 3D data with PCA pre-processing, it stores info about PCA.


Post-Lasso CLS algorithm
~~~~~~~~~~~~~~~~~~~~~~~~

This algorithms consists in applying CLS to restore the data and
determine the data support in DCT basis. A post-least square
optimization is performed to reduce the coefficients bias.

The method to give to the :meth:`~.signals.AbstractStem.restore` method is :code:`Post_LS_CLS`. The associated function is resp. :func:`~.restore.LS_CLS.Post_LS_CLS`.

The input parameters are:

* :code:`Lambda`: (float) The :math:`\lambda` regularization parameter,
* :code:`init`: (optional, numpy array) An initial point for the gradient descent algorithm which should have the same shape as the input data,
* :code:`Nit`: (optional, int) The number of iterations.

The output dictionary stores the following informations:

* :code:`E_CLS`: The evolution of the functional value for the CLS optimization step,
* :code:`E_post_ls`: The evolution of the functional value for the post-LS optimization step,
* :code:`Gamma`: The set of all pixel positions which coefficient in the DCT basis is non-zero,
* :code:`nnz-ratio`: The ratio of non-zero coefficients over the number of DCT coefficients,
* :code:`time`: the execution time (in sec.),
* :code:`PCA_info`: in case of 3D data with PCA pre-processing, it stores info about PCA.


ITKrMM and wKSVD
~~~~~~~~~~~~~~~~

Weighted K-SVD (see :cite:`b-mairal2008sparse`) and Iterative Thresholding and K residual Means for Masked data (see :cite:`b-naumova2018fast`) methods.

The wKSVD and ITKrMM algorithms share a lots of their code so that their input and output are the same. Though, two implementations exist to run these algorithms: one with python (:code:`ITKrMM` and :code:`wKSVD` methods) and one with maltab (:code:`ITKrMM_matlab` and :code:`wKSVD_matlab` methods). The original Matlab codes are broadcasted by `Karin Schnass`_. They were translated afterwards into python. Nothing distinguish them but for wKSVD where matlab is faster. The only problem is that you should have the :code:`matlab` command in your system path.

.. _Karin Schnass: https://www.uibk.ac.at/mathematik/personal/schnass/

The methods to give to the :meth:`~.signals.AbstractStem.restore` method are :code:`ITKrMM`, :code:`wKSVD`, :code:`ITKrMM_matlab` or :code:`wKSVD_matlab`. The associated functions are resp. :func:`~.restore.DL_ITKrMM.ITKrMM`, :func:`~.restore.DL_ITKrMM.wKSVD`, :func:`~.restore.DL_ITKrMM.ITKrMM_matlab` and :func:`~.restore.DL_ITKrMM.wKSVD_matlab`.

The input parameters are:

* :code:`Patchsize`: (optional, int) The patch width,
* :code:`K`: (optional, int) The dictionary size (incl. low-rank component),
* :code:`L`: (optional, int) The number of low-rank components to estimate,
* :code:`S`: (optional, int) The sparsity level,
* :code:`Nit`: (optional, int) The number of iterations for the dictionary estimation.
* :code:`Nit_lr`: (optional, int) The number of iterations for the low-rank estimation.

The output dictionary stores the following informations:

* :code:`dico`: The dictionary,
* :code:`E`: The evolution of the error,
* :code:`time`: the execution time (in sec.),
* :code:`PCA_info`: in case of 3D data with PCA pre-processing, it stores info about PCA.


BPFA
~~~~

Beta Process Factor Analysis algorithm (see :cite:`b-xing2012siam`).

As for wKSVD and ITKrMM, BPFA is based on a Matlab code from `Zhengming Xing`_ (these codes were broadcasted without any license). The python code just calls it, so matlab should be in the path system so that the :code:`matlab` command could be called from the command line.

.. _Zhengming Xing: https://zmxing.github.io/

The method to give to the :meth:`~.signals.AbstractStem.restore` method is :code:`BPFA_matlab`. The associated function is resp. :func:`~.restore.DL_BPFA.BPFA_matlab`.

The input parameters are:

* :code:`Patchsize`: (optional, int) The patch width,
* :code:`K`: (optional, int) The dictionary size,
* :code:`step`: (optional, int) That's the pixel space between two consecutive patches (if 1, full overlap),
* :code:`Nit`: (optional, int) The number of iterations for the dictionary estimation.

The output dictionary stores the following informations:

* :code:`dico`: The dictionary,
* :code:`time`: the execution time (in sec.),
* :code:`PCA_info`: in case of 3D data with PCA pre-processing, it stores info about PCA.


That's all folks !
------------------

This was the main content of the documentation. Congrats, you understood 90% of this library :)


References
----------

.. bibliography:: ../_static/references.bib
    :labelprefix: B
    :keyprefix: b-
