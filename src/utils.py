import numpy as np
import math
import logging


def get_logger():
    LOG = logging.getLogger(os.path.basename(__file__))
    LOG.setLevel(logging.DEBUG)

    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)
    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    LOG.addHandler(c_handler)
    LOG.warning("This is a warning")
    LOG.error("This is an error")
    return LOG

def calc_gencost(cof, qty, gen_id):
    return cof[gen_id, 0] * qty**2 + cof[gen_id, 1] * qty + cof[gen_id, 2]

def process_load_gen(daily_load_data, daily_renew_gen_data):
    """scale load, renewable_gen

    Args:
        daily_load_data (np.array): daily load data
        daily_renew_gen_data (np.array): daily renew data

    Returns:
        List[np.array]: load, wind, sol
    """
    ave_load = (np.max(daily_load_data) + np.min(daily_load_data)) / 2
    daily_load = (daily_load_data - ave_load) / ave_load + 1

    daily_wind_gen_data = daily_renew_gen_data[:, 0]
    daily_sol_gen_data = daily_renew_gen_data[:, 1]

    daily_wind_gen = daily_wind_gen_data / np.max(daily_wind_gen_data)
    daily_sol_gen = daily_sol_gen_data / np.max(daily_sol_gen_data)
    return daily_load, daily_wind_gen, daily_sol_gen


def sample_sol_wind_gen(wind_gen, sol_gen, MAX_NEW_LOAD, WEIBUL_PAR):
    """sample solar and wind gen from their distributions
    without timestep

    Assuming that wind, sol are scaled to [0, 1]

    Args:
        wind_gen (float): wind gen
        sol_gen (float): solar gen
        MAX_NEW_LOAD (float): max renewable load
        WEIBUL_PAR (float): coef

    Returns:
        Tuple[float, float, float]: sol, wind gen1, wind gen2
    """
    wind_gen *= MAX_NEW_LOAD
    sol_gen *= MAX_NEW_LOAD
    wind_gen1 = wind_gen * np.random.weibull(WEIBUL_PAR)
    wind_gen2 = wind_gen * np.random.weibull(WEIBUL_PAR)

    # 20%limit
    wind_gen1 = min(wind_gen * 1.2, wind_gen1, MAX_NEW_LOAD)
    wind_gen1 = max(wind_gen * 0.8, wind_gen1)

    wind_gen2 = min(wind_gen * 1.2, wind_gen2, MAX_NEW_LOAD)
    wind_gen2 = max(wind_gen * 0.8, wind_gen2)

    sol_gen = min(sol_gen * 1.2, sol_gen, MAX_NEW_LOAD)
    sol_gen = max(sol_gen * 0.8, sol_gen)
    return sol_gen, wind_gen1, wind_gen2


def sample_sol_wind_timestep(
    daily_load, daily_wind_gen, daily_sol_gen, timestep, MAX_NEW_LOAD, WEIBUL_PAR
):
    now_load = daily_load[timestep]
    now_wind_gen = daily_wind_gen[timestep] * MAX_NEW_LOAD
    now_sol_gen = daily_sol_gen[timestep] * MAX_NEW_LOAD

    # renew gen real
    now_wind_gen1 = now_wind_gen * np.random.weibull(WEIBUL_PAR)
    now_wind_gen2 = now_wind_gen * np.random.weibull(WEIBUL_PAR)

    # 20%limit
    now_wind_gen1 = min(now_wind_gen * 1.2, now_wind_gen1, MAX_NEW_LOAD)
    now_wind_gen1 = max(now_wind_gen * 0.8, now_wind_gen1)

    now_wind_gen2 = min(now_wind_gen * 1.2, now_wind_gen2, MAX_NEW_LOAD)
    now_wind_gen2 = max(now_wind_gen * 0.8, now_wind_gen2)

    now_sol_gen = min(now_sol_gen * 1.2, now_sol_gen, MAX_NEW_LOAD)
    now_sol_gen = max(now_sol_gen * 0.8, now_sol_gen)

    return now_load, now_sol_gen, now_wind_gen1, now_wind_gen2


def generate_piecewise_price(gmax, cof, gen_id, action=1.0):
    # 4 piecewise bidding
    n_piecewise = 4

    # avoid inplace changes on np.array
    tmp_cof = np.copy(cof)
    tmp_cof[gen_id, :] = cof[gen_id, :] * action
    n_gen = cof.shape[0]

    # price, vol pairs
    gencost = np.zeros((n_gen, n_piecewise * 2))
    for i in range(n_gen):
        cof1 = tmp_cof[i, 0]
        cof2 = tmp_cof[i, 1]
        cof3 = tmp_cof[i, 2]
        gen_gmax = gmax[i]
        for j in range(n_piecewise):
            cur_vol = gen_gmax / (n_piecewise - 1) * j
            cur_pri = cof1 * cur_vol**2 + cof2 * cur_vol + cof3
            gencost[i, 2 * j] = cur_vol
            gencost[i, 2 * j + 1] = cur_pri
    offers_qty = np.zeros((n_gen, n_piecewise - 1))
    offers_prc = np.zeros((n_gen, n_piecewise - 1))
    for i in range(n_piecewise - 1):
        offers_qty[:, i] = gencost[:, (i + 1) * 2] - gencost[:, i * 2]
        offers_prc[:, i] = (
            gencost[:, (i + 1) * 2 + 1] - gencost[:, i * 2 + 1]
        ) / offers_qty[:, i]
    return offers_qty, offers_prc, gencost


def calc_gencost(cof, qty, gen_id):
    """calculate gencost based on cof

    Args:
        cof (np.ndarray): cost coef of gen
        qty (float): quantity of gen
        gen_id (int): id of gen

    Returns:
        cost (np.array)
    """
    return cof[gen_id, 0] * qty**2 + cof[gen_id, 1] * qty + cof[gen_id, 2]


def calc_profit_cvar(cof, result, gen_id):
    """calculate profit based on priceSim result

    Args:
        cof (np.ndarray): cost coef of gen
        result (np.ndarray): result of priceSim, column 0 is qty, column 1 is price,
                             column 3 is gencost. However, we use the gencost based on cof
        gen_id (int): id of gen

    Returns:
        profit (np.array)
    """
    cost = calc_gencost(cof, result[:, 0], gen_id)
    profit = result[:, 0] * result[:, 1] - cost
    first_n = math.floor(profit.shape[0] // 10)
    cvar = np.sort(profit)[:first_n].mean(0)
    return profit, cvar
