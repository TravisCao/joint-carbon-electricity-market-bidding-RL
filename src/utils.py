import numpy as np
import math
import logging
import os


def get_logger():
    """get logger set up

    Returns:
        logging.Logger: logger
    """
    LOG = logging.getLogger(os.path.basename(__file__))
    LOG.setLevel(logging.DEBUG)

    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)
    c_format = logging.Formatter("%(name)s - %(levelname)s: %(message)s")
    c_handler.setFormatter(c_format)
    LOG.addHandler(c_handler)
    return LOG