#!/usr/bin/env python3

"""Basic logging module."""

import logging

DEFAULT_LOGGER_STRING_FORMAT = '%(asctime)s %(levelname)-8s' \
    '[%(filename)s:%(lineno)d] %(message)s'


def setup_logger(name, log_file):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(DEFAULT_LOGGER_STRING_FORMAT)
    logger.addHandler(handler)

    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.WARN)
    handler.setFormatter(DEFAULT_LOGGER_STRING_FORMAT)
    logger.addHandler(handler)

    return logger


log = setup_logger('log', log_file='warn.log')
