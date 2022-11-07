import numpy as np


class Config:
    LOAD_COEF = 1.05
    MAX_NEW_LOAD = 20
    N_EPOCH = 50
    EPOCH_START = 0

    # distribution coef for wind
    WEIBUL_PAR = 4.8

    # distribution coef for solar
    BETA_PAR1 = 4
    BETA_PAR2 = 2


    # select gen_id = 1 because in select_gen.ipynb
    # the action for optimal cvar and optimal profit is different
    agent_gen_id = 1

    eval_frequency = 10
    n_samples = 500
    n_action_centers = 20
    n_clusters = 8


    cof = np.array(
        [
            [0.5900, 9.0000, 0],
            [0.6875, 8.500, 0],
            [0.7900, 7.5000, 0],
            [0.8975, 8.000, 0],
            [0.7120, 7.5000, 0],
            [0.7900, 8.0000, 0],
        ]
    )

    gmax_fn = lambda _, now_sol_gen, now_wind_gen1: np.array(
        [60, 60, 60, 60, 60, 60, now_sol_gen, now_wind_gen1]
    )
