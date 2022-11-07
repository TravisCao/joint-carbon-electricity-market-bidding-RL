from typing import List, Tuple
import numpy as np
import pandas as pd
import matlab
import math
from tqdm import tqdm
from config import Config
from utils import get_logger


## electricity market
class ElectricityMarket:
    """electricity market"""

    def __init__(
        self,
        config: Config,
        engine: matlab.engine,
        load_data_path: str,
        renew_data_path: str,
    ) -> None:
        """
        Args:
            config (Config): configuration file
            engine (matlab.engine): matlab engine
            load_data_path (str): load data path (txt file)
            renew_data_path (str): renew data path (txt file)
        """

        self.config = config
        self.cof = config.cof
        self.agent_gen_id = config.agent_gen_id
        self.engine = engine
        # load in each timestep
        self.loads = np.loadtxt(load_data_path) * self.LOAD_COEF
        # renews (wind and solar) in each timestep
        self.renews = np.loadtxt(renew_data_path)

        self.LOAD_COEF = config.LOAD_COEF
        self.MAX_NEW_LOAD = config.MAX_NEW_LOAD
        self.WEIBUL_PAR = config.WEIBUL_PAR
        self.BETA_PAR1 = config.BETA_PAR1
        self.BETA_PAR2 = config.BETA_PAR2

        self.timestep = 0
        self.LOG = get_logger()

    def reset_timestep(self):
        self.timestep = 0

    def run_step(
        self,
        wind: float,
        solar: float,
        load: float,
        offers_qty: np.ndarray,
        offers_prc: np.ndarray,
    ) -> np.ndarray:
        """run electricity market in one timestep (hourly or half an hour)
        assumes that one wind generator and one solar generator

        Args:
            wind (float): wind power
            solar (float): solar power
            load (float): load coefficient for all load buses
            offers_qty (np.ndarray): Pg quantities in offer
            offers_prc (np.ndarray): Pg price in offer
        """
        result = self.enging.price_sim(
            matlab.double([load]),
            matlab.double([solar]),
            matlab.double([wind]),
            matlab.double([self.MAX_NEW_LOAD]),
            matlab.double(offers_qty.tolist()),
            matlab.double(offers_prc.tolist()),
        )

        if not result["success"]:
            self.LOG.error(f"run market not success!")
            return
        else:
            return np.array(result["clear"][self.agent_gen_id])

    @staticmethod
    def calc_gencost(cof: np.ndarray, qty: np.array, gen_id: int) -> float:
        """calculate cost of a gen

        Args:
            cof (np.ndarray): gencost (polynomial) coef
            qty (np.array): gen quantity
            gen_id (int): gen id

        Returns:
            float: overall cost of the gen
        """
        return cof[gen_id, 0] * qty**2 + cof[gen_id, 1] * qty + cof[gen_id, 2]

    def process_load_gen(self) -> List[np.ndarray, np.ndarray, np.ndarray]:
        """scale load, renewable_gen

        Returns:
            List[np.array]: load, wind, sol
        """
        mean_load = np.mean(self.loads)
        loads = (self.loads - mean_load) / mean_load + 1

        wind_gen = self.renews[:, 0]
        solar_gen = self.renews[:, 1]

        wind_gen = wind_gen / np.max(wind_gen)
        sol_gen = solar_gen / np.max(solar_gen)
        return loads, wind_gen, sol_gen

    def sample_sol_wind_gen_step(
        self,
        wind: float,
        solar: float,
    ) -> List[float, float, float]:
        """sample solar and wind gen in a step

        Args:
            wind (float): wind expectation
            solar (float): solar expectation

        Returns:
            List[float, float, float]: solar power, wind power1, wind power2
        """
        wind_gen = wind * self.MAX_NEW_LOAD
        sol_gen = solar * self.MAX_NEW_LOAD

        # renew gen real
        wind_gen1 = wind_gen * np.random.weibull(self.WEIBUL_PAR)
        wind_gen2 = wind_gen * np.random.weibull(self.WEIBUL_PAR)

        sol_gen = sol_gen * np.random.beta(self.BETA_PAR1, self.BETA_PAR2)

        # 20%limit
        wind_gen1 = min(wind_gen * 1.2, wind_gen1, self.MAX_NEW_LOAD)
        wind_gen1 = max(wind_gen * 0.8, wind_gen1)

        wind_gen2 = min(wind_gen * 1.2, wind_gen2, self.MAX_NEW_LOAD)
        wind_gen2 = max(wind_gen * 0.8, wind_gen2)

        sol_gen = min(sol_gen * 1.2, sol_gen, self.MAX_NEW_LOAD)
        sol_gen = max(sol_gen * 0.8, sol_gen)
        return sol_gen, wind_gen1, wind_gen2

    def generate_piecewise_price(
        self, gmax: list, cof: np.ndarray, gen_action: float = 1.0
    ) -> List[np.ndarray, np.ndarray, np.ndarray]:
        """generate bidding strategy (price, vol) pair and gencost

        Args:
            gmax (list): maximum Pg of each gen
            cof (np.ndarray): cof of gen cost
            gen_action (float): agent_gen action (overall coefficient in gencost fn). Defaults to 1.0

        Returns:
            List[np.ndarray, np.ndarray, np.ndarray]: quantities in offer, prices in offer, qty_prc_pairs
        """

        # 4 piecewise bidding
        n_piecewise = 4

        # avoid inplace changes on np.array
        tmp_cof = np.copy(cof)
        tmp_cof[self.agent_gen_id, :] = cof[self.agent_gen_id, :] * gen_action
        n_gen = cof.shape[0]

        # price, vol pairs
        qty_prc_pairs = np.zeros((n_gen, n_piecewise * 2))
        for i in range(n_gen):
            cof1 = tmp_cof[i, 0]
            cof2 = tmp_cof[i, 1]
            cof3 = tmp_cof[i, 2]
            gen_gmax = gmax[i]
            for j in range(n_piecewise):
                cur_vol = gen_gmax / (n_piecewise - 1) * j
                cur_pri = cof1 * cur_vol**2 + cof2 * cur_vol + cof3
                qty_prc_pairs[i, 2 * j] = cur_vol
                qty_prc_pairs[i, 2 * j + 1] = cur_pri
        offers_qty = np.zeros((n_gen, n_piecewise - 1))
        offers_prc = np.zeros((n_gen, n_piecewise - 1))
        for i in range(n_piecewise - 1):
            offers_qty[:, i] = qty_prc_pairs[:, (i + 1) * 2] - qty_prc_pairs[:, i * 2]
            offers_prc[:, i] = (
                qty_prc_pairs[:, (i + 1) * 2 + 1] - qty_prc_pairs[:, i * 2 + 1]
            ) / offers_qty[:, i]
        return offers_qty, offers_prc, qty_prc_pairs

    def calc_profit_cvar(
        self, cof: np.ndarray, result: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """calculate profit of the agent gen based on price_sim result

        Args:
            cof (np.ndarray): cost coef of gen
            result (np.ndarray): results of multiple price_sim steps, column 0 is qty,
                                 column 1 is price, column 3 is gencost.
                                 However, we use the gencost based on coef

        Returns:
        """
        cost = self.calc_gencost(cof, result[:, 0], self.agent_gen_id)
        profit = result[:, 0] * result[:, 1] - cost
        first_n = math.floor(profit.shape[0] // 10)
        cvar = np.sort(profit)[:first_n].mean(0)
        return profit, cvar
