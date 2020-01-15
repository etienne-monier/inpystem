# -*- coding: utf-8 -*-
"""
This small module defines some tools to handle messages sent to the
console. More specially, it implements three functions which 
"""

LEVEL = 0


def c_welc_print(message):
    """Prints a welcome message to console.

    Arguments
    ---------
    message: str
        The welcome message.
    """
    global LEVEL
    LEVEL += 1

    print('\t' * LEVEL + '--- ' + message + ' ---')


def c_print(message):
    """Prints a message to console.

    Arguments
    ---------
    message: str
        The message.
    """
    for text in message.split('\n'):
        print('\t' * LEVEL + text)


def c_end_print(message):
    """Prints a closing message to console.

    Arguments
    ---------
    message: str
        The closing message.
    """
    global LEVEL
    c_print(message)
    c_print('---')

    LEVEL -= 1
