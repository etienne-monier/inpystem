# -*- coding: utf-8 -*-

import logging
import logging.handlers

import pathlib


def get_log_file_location():
    """Get the path to log file.

    Returns
    -------
    str
        The log file location.
    """
    p = pathlib.Path(__file__)
    return str(p.parent) + '/inpystem.log'


def configure_logger():
    """Configures the logger.
    """
    console_format = '%(levelname)s - %(name)s - %(message)s"'
    logging.basicConfig(
        level=logging.WARNING,
        format=console_format,
        )

    _logger = logging.getLogger('inpystem')
    if not _logger.handlers:

        # Define a Handler which writes INFO messages or higher to a
        # log file
        file = logging.handlers.RotatingFileHandler(
            get_log_file_location(),
            maxBytes=1e6,
            backupCount=0,
            encoding='utf8')

        # Set level.
        file.setLevel(logging.DEBUG)

        # set a format which is simpler for file use
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(format_str)

        # tell the handler to use this format
        file.setFormatter(formatter)

        # add the handler to the root logger
        _logger.addHandler(file)


def set_log_level(level):
    """
    Convenience function to set the log level of all inpystem modules.
    Note: The log level of all other modules are left untouched.

    Parameters
    ----------
    level : {int | str}
        The log level to set. Any values that `logging.Logger.setLevel()`
        accepts are valid. The default options are:
            - 'CRITICAL'
            - 'ERROR'
            - 'WARNING'
            - 'INFO'
            - 'DEBUG'
            - 'NOTSET'
    """
    logging.basicConfig()  # Does nothing if already configured
    logging.getLogger('inpystem').setLevel(level)
