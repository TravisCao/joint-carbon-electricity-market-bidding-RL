import numpy as np
import os


class Config:
    LOAD_COEF = 1.05
    MAX_NEW_LOAD = 20

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

    # max gen for non-renewable gens
    gmax = 60

    n_timesteps = 48

    gen_units = ["KS_Jeffrey Energy Center_unit2_Coal_P733" for _ in range(n_gens)]

    gencost_coef = np.array(
        [
            [0.5900, 9.0000, 0],
            [0.6875, 8.500, 0],
            [0.7900, 7.5000, 0],
            [0.8975, 8.000, 0],
            [0.7120, 7.5000, 0],
            [0.7900, 8.0000, 0],
        ]
    )

    # data
    config_file_path = os.path.dirname(os.path.abspath(__file__))
    load_data_path = f"{config_file_path}/../data/load.txt"
    renew_data_path = f"{config_file_path}/../data/30min.txt"
    cems_data_path = f"{config_file_path}/../data/cems_coef.csv"

    # carbon market
    n_trading_days = 100

    # free carbon allowance 90%, expected 46521 * 100 / 6 (n_gens) * 0.9
    carbon_allowance_initial = 697815
    carbon_price_initial = 10

    # TODO: check if this needs to be changed
    # settings in "A hybrid interactive simulation method for studying emission trading behaviors"
    price_alpha = 10
    price_beta = 2
