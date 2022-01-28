from .log import get_logger
from MGB.Modules import Modules
M = Modules()


def timer(func):
    """Timer decorator
    Time the execution time of the passed in function.
    """

    @M.functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Create logger object
        logger = get_logger(
            log_file_name=self.log_file_name, log_dir=self.log_file_dir)

        t1 = M.time.time()
        result = func(self, *args, **kwargs)
        t2 = M.time.time() - t1
        logger.info("|- " + str(func.__name__) + " was executed in: " +
                    str(t2)[:5] + " [sec(s)]")
        return result

    return wrapper


def logger(func):
    """Control print messages

    Takes care of both the dumped .log file
    and the console ouput. The rule used here is as follows:
    - Only general and DEBUG information are log inside the decorator.
    - Anything else is logged inside the funciton being called.

    The hierarchy is:
    LEVEL      NUMERIC VALUE
    CRITICAL : 50
    ERROR    : 40
    WARNING  : 30
    INFO     : 20
    DEBUG    : 10
    NOTSET   : 0 
    """

    @M.functools.wraps(func)
    def wrapper(self, *args, **kwargs):

        # Create logger object
        logger_obj = get_logger(
            log_file_name=self.log_file_name, log_dir=self.log_file_dir)

        # This is a series of general info applicable to any methods/functions
        logger_obj.debug("|")
        logger_obj.debug("|- Enter method")
        logger_obj.debug("|- Method's name: " + func.__name__)
        logger_obj.debug("|- Method's args: {}".format(args))
        logger_obj.debug("|- Method's kwargs: {}".format(kwargs))

        # Call the function as usual
        value = func(self, *args, **kwargs)
        logger_obj.debug("|- Exit method")
        logger_obj.debug("|")

        return value

    return wrapper
