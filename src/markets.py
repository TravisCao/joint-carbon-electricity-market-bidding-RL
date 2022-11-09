"""
Project: src
File Created: Wednesday, 9th November 2022 10:57:53 am
Author: Yuji Cao (travisyjcao@gmail.com)
-----
Last Modified: Wednesday, November 9th 2022, 12:34:45 pm
Modified By: Yuji Cao
"""

import math
from typing import Tuple, List

import matlab.engine
import numpy as np
import pandas as pd

from config import Config
from utils import get_logger
from gencon import GenCon

## electricity market
class ElectricityMarket:
    """electricity market"""

    def __init__(
        self,
        gens: List[GenCon],
        config: Config,
        engine: matlab.engine,
    ) -> None:
        """
        Args:
            config (Config): configuration file
            engine (matlab.engine): matlab engine
            load_data_path (str): load data path (txt file)
            renew_data_path (str): renew data path (txt file)
        """
        self.engine = engine
        self.config = config
        self.LOAD_COEF = config.LOAD_COEF
        self.MAX_NEW_LOAD = config.MAX_NEW_LOAD
        self.WEIBUL_PAR = config.WEIBUL_PAR
        self.BETA_PAR1 = config.BETA_PAR1
        self.BETA_PAR2 = config.BETA_PAR2
        self.gencost_coef = config.gencost_coef  # gen cost in generation
        self.agent_gen_id = config.agent_gen_id
        self.n_gens = config.n_gens
        self.gen_types = config.gen_types  # non-renewable gen types

        # load in each timestep
        self.loads = np.loadtxt(config.load_data_path) * self.LOAD_COEF
        # renews (wind and solar) in each timestep
        self.renews_exp = np.loadtxt(config.renew_data_path)
        # scaled loads, wind Pg exp., solar Pg exp.
        self.loads, self.wind_gen_exps, self.solar_gen_exps = self.process_load_gen()

        self.timestep = 0

        # column index in run result
        self.PRC_COL = 1
        self.QTY_COL = 0

        self.LOG = get_logger()

    def reset_timestep(self):
        """reset timestep"""
        self.timestep = 0

    def increase_timestep(self):
        """increase timestep and reset"""
        self.timestep += 1
        if self.timestep == len(self.loads):
            self.reset_timestep()

    def run_step(self, gen_action):
        """run eletricity market in one timestep

        Args:
            gen_action (float): gen action coef (bidding strategy)

        Returns:
            result (np.ndarray): market clearing result of agent_gen_id
                                col 0: quantity
                                col 1: price
                                col 2: cost
        """
        load_step = self.loads[self.timestep]
        wind_exp_step = self.wind_gen_exps[self.timestep]
        solar_exp_step = self.solar_gen_exps[self.timestep]

        # sample solar wind in this step
        solar_gen_step, wind_gen1_step, wind_gen2_step = self.sample_sol_wind_gen_step(
            wind_exp_step, solar_exp_step
        )
        gmax = self.config.gmax_fn(solar_gen_step, wind_gen1_step)
        offers_qty, offers_prc, qty_prc_pairs = self.generate_piecewise_price(
            gmax, self.gencost_coef, gen_action
        )

        result = self._run_step(
            wind_gen1_step, solar_gen_step, load_step, offers_qty, offers_prc
        )

        self.increase_timestep()
        return result

    def _run_step(
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
        result = self.engine.price_sim(
            matlab.double([load]),
            matlab.double([solar]),
            matlab.double([wind]),
            matlab.double([self.MAX_NEW_LOAD]),
            matlab.double(offers_qty.tolist()),
            matlab.double(offers_prc.tolist()),
        )

        if not result["success"]:
            self.LOG.warning("run market not success!")
            return
        else:
            return np.array(result["clear"])

    def process_load_gen(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """scale load, renewable_gen

        Returns:
            Tuple[np.array]: scaled loads, wind Pg expectation, solar Pg expactation
        """
        mean_load = np.mean(self.loads)
        loads = (self.loads - mean_load) / mean_load + 1

        wind_gen_exps = self.renews_exp[:, 0]
        solar_gen_exps = self.renews_exp[:, 1]

        wind_gen_exps = wind_gen_exps / np.max(wind_gen_exps)
        sol_gen = solar_gen_exps / np.max(solar_gen_exps)
        return loads, wind_gen_exps, sol_gen

    def sample_sol_wind_gen_step(
        self,
        wind: float,
        solar: float,
    ) -> Tuple[float, float, float]:
        """sample solar and wind gen in a step

        Args:
            wind (float): wind expectation
            solar (float): solar expectation

        Returns:
            Tuple[float, float, float]: solar power, wind power1, wind power2
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """generate bidding strategy (price, vol) pair and gencost

        Args:
            gmax (list): maximum Pg of each gen
            cof (np.ndarray): cof of gen cost
            gen_action (float): agent_gen action (overall coefficient in gencost fn).
                                Defaults to 1.0

        Returns:
            offers_qty(np.ndarray): quantities in offer
            offers_prc(np.ndarray): prices in offer
            qty_prc_pairs(np.ndarray): quantity price pairs
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

    def calc_gencost(self, qty: np.array) -> float:
        """calculate costs of all gens, one or multiply runs

        Args:
            cof (np.ndarray): gencost (polynomial) coef
            qty (np.array): gen quantity in one/multiple runs

        Returns:
            float: overall cost of the gen
        """
        return (
            self.gencost_coef[:, 0] * qty**2
            + self.gencost_coef[:, 1] * qty
            + self.gencost_coef[:, 2]
        )

    def calc_agent_gen_cvar(
        self, clear_result_gen: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """calculate cvar of the agent gen based on price_sim result in multiple runs

        assumes that clear_result_gen is the result of one gen in multiple runs

        cost function based on self.cof

        Args:
            clear_result_gen (np.ndarray): results of multiple price_sim steps
                                 we use the gencost based on polynomial

        Returns:
            profit (np.ndarray): profits in multiple runs
            cvar (float): cvar of the gen in multiple runs
        """
        profit = self.calc_gen_profit(clear_result_gen)
        first_n = math.floor(profit.shape[0] // 10)
        cvar = np.sort(profit)[:first_n].mean(0)
        return cvar

    def calc_gen_profit(
        self,
        clear_result: np.ndarray,
    ) -> np.ndarray:
        """calculate profits of all gens or one gen in multiple runs

        Args:
            clear_result (np.ndarray): clearing result of one price_sim run

        Returns:
            np.ndarray: _description_
        """
        costs = self.calc_gencost(clear_result[:, self.QTY_COL])
        return clear_result[:, self.QTY_COL] * clear_result[:, self.PRC_COL] - costs


class CarbonMarket:
    """
    Carbon market
    """

    def __init__(self, config) -> None:

        self.carbon_price = 0.0
        self.carbon_quota = 0.0
        self.day_t = 0


if __name__ == "__main__":
    engine = matlab.engine.start_matlab()
    gens = [GenCon(i, Config.gen_types[i], Config) for i in range(Config.n_gens)]
    elec_market = ElectricityMarket(
        gens,
        Config,
        engine,
    )
    res = elec_market.run_step(1.0)
    elec_market.calc_carbon_emission(res)
