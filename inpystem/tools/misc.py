# -*- coding: utf-8 -*-
"""This module defines some miscellaneous functions.
"""


def toslice(text=None, length=None):
    """Parses a string into a slice.

    Input strings can be eg. '5:10', ':10', '1:'. Negative limits are
    allowed only if the data length is given. In such case, input
    strings can be e.g. '1:-10'.

    If no text not length is given, default slice is slice(None).

    Arguments
    ---------
    text: optional, None, str
        The input text to parse.
        Default is None.
    length: None, int
        The data length. This is not mendatory if no slice limit
        is negative.
        Dafault is None.

    Returns
    -------
    slice
        The parsed slice object.
    """

    # For default input.
    if text is None:
        return slice(None)

    # Getting slice limits and storing it into lim.
    lim = text.split(':')

    for cnt in range(2):
        lim[cnt] = None if lim[cnt] == '' else eval(lim[cnt])

    # Let us chack that length is given in case limits are negative
    if ((lim[0] is not None and lim[0] < 0) or
            (lim[1] is not None and lim[1] < 0)) and length is None:
        raise ValueError(
            'Please give the length argument to handle negative limits.')

    # The non-None limits are transformed if length is given to
    # avoid negative or 'greater than length' limits.
    for cnt in range(2):
        if lim[cnt] is not None and lim[cnt] < 0 and length is not None:
            lim[cnt] = lim[cnt] % length

    # Last check before output: if limits are a:b, does b is really
    # greater than a ?
    if None not in lim and lim[0] >= lim[1]:
        raise ValueError(
            'The slice lower bound is greater or equal to the '
            'slice upper bound.')

    return slice(*lim)
