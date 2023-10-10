import numpy as np
import os
import gymnasium as gym


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

    n_timesteps = 24

    num_mkt = 10

    PRC_COL = 1
    QTY_COL = 0

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
    load_data_path = f"{config_file_path}/../data/load-1h.txt"
    renew_data_path = f"{config_file_path}/../data/1h.txt"
    cems_data_path = f"{config_file_path}/../data/cems_coef.csv"

    # carbon market
    n_trading_days = 10

    # free carbon allowance 90%, expected 46521 * 100 / 6 (n_gens) * 0.9
    carbon_price_initial = 40
    carbon_allowance_initial = 3500 * n_trading_days

    carbon_penalty = 1e8

    # TODO: check if this needs to be changed
    # settings in "A hybrid interactive simulation method for studying emission trading behaviors"
    price_alpha = 10
    price_beta = 2

    elec_obs_dim = 4
    elec_act_dim = 1

    carb_obs_dim = 5
    carb_act_dim = 1

    elec_obs_space = gym.spaces.Box(
        low=0, high=np.inf, shape=(elec_obs_dim,), dtype=np.float32
    )
    elec_act_space = gym.spaces.Box(low=1.0, high=2.0, dtype=np.float32)

    elec_act_high = 2.0
    elec_act_low = 2.0

    reward_scale = 3e-3

    total_timesteps = int(1e6)

    # ddpg args
    # seed = 1
    # torch_deterministic = False
    # cuda = True
    # track = True
    # wandb_project_name = "MARL-joint-carbon-electricity-market"

    # save_model = True
    # upload_model = False
