import os
import logging


def create_logger(logger_name, log_path = None, append = False):
    
    if not append and os.path.isfile(log_path):
        with open(log_path, 'w') as f: pass
    # If append is False and the file exists, clear the content of the file.

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    if log_path is None:
        handler = logging.StreamHandler() # show log in console
    else:
        handler = logging.FileHandler(log_path) # print log in file
    
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            fmt = '%(asctime)s %(levelname)s:  %(message)s',
            datefmt ='%m-%d %H:%M'
        )
    )
    logger.addHandler(handler)

    return logger