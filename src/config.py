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

    # 6 non-renewable gens in t_auction_case
    n_gens = 6
    # gen_types = [None for _ in range(n_gens)]
    gen_types = [
        "CA_Valley Generating Station_unit5_P47",
        "CO_Brush Power Projects_unitGT5_P79",
        "CO_Pueblo Airport Generating Station_unitCT02_P94",
        "GA_Dahlberg (Jackson County)_unit1_P92",
        "GA_Dahlberg (Jackson County)_unit3_P93",
        "GA_Dahlberg (Jackson County)_unit5_P92",
    ]

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

    gmax_fn = lambda sol_gen_step, wind_gen1_step: np.array(
        [60, 60, 60, 60, 60, 60, sol_gen_step, wind_gen1_step]
    )

    # data
    load_data_path = "../data/load.txt"
    renew_data_path = "../data/30min.txt"
    cems_data_path = "../data/cems_coef.csv"
