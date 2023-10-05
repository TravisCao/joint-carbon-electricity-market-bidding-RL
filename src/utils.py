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
    LOG.setLevel(logging.INFO)

    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_format = logging.Formatter("%(name)s - %(levelname)s: %(message)s")
    c_handler.setFormatter(c_format)
    LOG.addHandler(c_handler)
    return LOG


def calc_gen_reward(res, config):
    """calculate reward for a generator

    reward is price * quantity - cost

    Args:
        res (np.ndarray): market clearing result

    Returns:
        float: reward
    """
    r = (
        res[config.agent_gen_id, config.QTY_COL]
        * res[config.agent_gen_id, config.PRC_COL]
        - res[config.agent_gen_id, -1]
    )
    return r * config.reward_scale
