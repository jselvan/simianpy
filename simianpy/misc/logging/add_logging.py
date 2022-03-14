import functools

from simianpy.misc.logging.getlogger import getLogger

default_logger_kwargs = dict(loggerName="main")


def add_logging(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if "logger" in kwargs:
            logger = kwargs.pop("logger")
        else:
            if "logger_kwargs" in kwargs:
                logger_kwargs = kwargs.pop("logger_kwargs")
            else:
                logger_kwargs = default_logger_kwargs
            logger = getLogger(**logger_kwargs)
        return function(*args, **kwargs, logger=logger)

    return wrapper
