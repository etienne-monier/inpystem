#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This small module only contain sec2str, which is a function to display
time in human-readable format.
"""


def sec2str(t):
    """Returns a human-readable time str from a duration in s.

    Arguments
    ---------
    t: float
        Duration in seconds.

    Returns
    -------
    str
        Human-readable time str.

    Example
    -------
    >>> from inpystem.tools.sec2str import sec2str
    >>> sec2str(5.2056)
    5.21s
    >>> sec2str(3905)
    '1h 5m 5s'
    """

    # Decompose into hour, minute and seconds.
    h = int(t // 3600)
    m = int((t - 3600 * h) // 60)
    s = t - 3600 * h - 60 * m

    # Print digits if non-int seconds
    if isinstance(s, int):
        s_str = '{:d}'.format(s)
    else:
        s_str = '{:.2f}'.format(float(s))

    # Display info depending on available elements.
    if h == 0 and m == 0:
        return "{}s".format(s_str)

    elif h == 0 and m != 0:
        return "{:d}m {}s".format(m, s_str)

    else:
        return "{:d}h {:d}m {}s".format(h, m, s_str)
