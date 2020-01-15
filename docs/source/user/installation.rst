Installing inpystem
===================

With pip
--------

inpystem is listed in the `Python Package Index`_. Therefore, it can be automatically downloaded and installed with pip. You may need to install pip for the following commands to run.

.. _Python Package Index: https://pypi.org/

Install with pip:

.. code-block:: bash

    $ pip install inpystem

Install from source
-------------------

When installing manually, be sure that all dependences are required. For example, do:

.. code-block:: bash

    $ pip install numpy scipy matplotlib hyperspy scikit-learn scikit-image

Be aware that the scikit-image package versions is above 0.16.

Released version
~~~~~~~~~~~~~~~~

To install from source grab a tar.gz release from `Python Package Index <https://pypi.org/>` and use the  following code if Linux/Mac user:

.. code-block:: bash

    $ tar -xzf inpystem.tar.gz
    $ cd inpystem
    $ python setup.py install

You can also use a Python installer, e.g.

.. code-block:: bash

    $ pip install inpystem.tar.gz


Development version
~~~~~~~~~~~~~~~~~~~

To get the development version from our git repository you need to install git. Then just do:

.. code-block:: bash

    $ git clone https://github.com/etienne-monier/inpystem
    $ cd inpystem

Then, perform one of the following commands:

.. code-block:: bash

    $ pip install -e .
    $ python setup.py install





