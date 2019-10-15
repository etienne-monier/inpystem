# -*- coding: utf-8 -*-

import os

import numpy as np
import hyperspy.api as hs

from .. import signals


def path_from_file(data_file, ratio=1):
    """Creates a Path object  from a data file (such as .dm3, .dm4 or npz).

    In the case of a .npz file, this one should contain the size and path
    arrays.

    Concerning the .dm3/.dm4 files, the data storage is specific to the LPS
    Lab (Orsay, France) implementation.

    Arguments
    ---------
    data_file: str
        data file location.
    ratio: optional, float
        The ratio of sampled pixels. This should lay between 0 and 1.
        Default is 1.
    """
    # Get file extension
    _, file_ext = os.path.splitext(data_file)

    # Digital Micrograph file
    if (file_ext == '.dm3' or file_ext == '.dm4'):
        data = hs.load(data_file).data

        if data.ndim == 3:
            size = data.size[:2]
            x = data[:, :, 0].flatten()-1
            y = data[:, :, 1].flatten()-1

            path = x + y*size[1]

        elif data.ndim == 2:
            size = data.size
            path = data.flatten()

        else:
            raise ValueError(
                'Path data has {} dimensions instead of 2 or 3.'
                .format(data.ndim))

    # Numpy file
    elif (file_ext == '.npz'):
        data = np.load(data_file)

        x = data['x']
        y = data['y']
        size = tuple(data['size'])

        path = x + y*size[1]

    else:
        raise ValueError(
            'The path data file extension {} is not appropriate.'
            .format(file_ext))

    return signals.Path(size, path)
