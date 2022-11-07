import math
from typing import Tuple

import matlab.engine
import numpy as np
import pandas as pd

from config import Config
from utils import get_logger


## electricity market
class ElectricityMarket:
    """electricity market"""

    def __init__(
        self,
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
        self.cof = config.cof  # gen cost in generation
        self.agent_gen_id = config.agent_gen_id
        self.n_gens = config.n_gens
        self.gen_types = config.gen_types  # non-renewable gen types

        # load in each timestep
        self.loads = np.loadtxt(config.load_data_path) * self.LOAD_COEF
        # renews (wind and solar) in each timestep
        self.renews_exp = np.loadtxt(config.renew_data_path)
        # scaled loads, wind Pg exp., solar Pg exp.
        self.loads, self.wind_gen_exps, self.solar_gen_exps = self.process_load_gen()

        # index_col = 0 to ignore the extra index column
        self.cems_df = pd.read_csv(config.cems_data_path)

        self.timestep = 0
        self.LOG = get_logger()
        self.setup_gen_emission_coef()

    def setup_gen_emission_coef(self):
        """Set up gen type in cems"""
        gen_idxs = [
            self.cems_df[self.cems_df["unit"] == gen_type].index[0]
            for gen_type in self.gen_types
        ]
        self.gen_emission_coef_quad = self.cems_df.iloc[gen_idxs][
            ["coef-2-emission-quad", "coef-1-emission-quad", "coef-0-emission-quad"]
        ].values
        self.gen_emission_coef_linear = self.cems_df.iloc[gen_idxs][
            ["coef-1-emission-linear", "coef-0-emission-linear"]
        ].values
        self.gen_emission_split_point_x = self.cems_df.iloc[gen_idxs][
            "split-point-x"
        ].values

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
            gmax, self.cof, gen_action
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
            self.LOG.warning(f"run market not success!")
            return
        else:
            return np.array(result["clear"])

    @staticmethod
    def calc_gencost(cof: np.ndarray, qty: np.array, gen_id: int) -> float:
        """calculate cost of a gen, one or multiply steps

        Args:
            cof (np.ndarray): gencost (polynomial) coef
            qty (np.array): gen quantity
            gen_id (int): gen id

        Returns:
            float: overall cost of the gen
        """
        return cof[gen_id, 0] * qty**2 + cof[gen_id, 1] * qty + cof[gen_id, 2]

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

    def calc_profit_cvar(
        self, cof: np.ndarray, clear_result_gen: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """calculate profit of the agent gen based on price_sim result

        Args:
            cof (np.ndarray): cost coef of gen
            result (np.ndarray): results of multiple price_sim steps, column 0 is qty,
                                 column 1 is price, column 3 is gencost.
                                 However, we use the gencost based on coef

        Returns:
        """
        cost = self.calc_gencost(cof, clear_result_gen[:, 0], self.agent_gen_id)
        profit = clear_result_gen[:, 0] * clear_result_gen[:, 1] - cost
        first_n = math.floor(profit.shape[0] // 10)
        cvar = np.sort(profit)[:first_n].mean(0)
        return profit, cvar

    def calc_carbon_emission(self, clear_result: np.ndarray) -> np.ndarray:
        """calculate carbon emission of gens based on market clearing result
        the carbon emission is divided into quadratic and linear parts
        quadratic: gen emission under the split point x
        linear: gen emission over the split point x

        Args:
            clear_result (np.ndarray): market clearing result

        Returns:
            emissions(np.ndarray): carbon emission of each gen
        """
        gen_clear_qty = clear_result[:, 0]

        # linear emission for gen quantity over the split point
        gen_qty_linear_emission = np.maximum(
            gen_clear_qty - self.gen_emission_split_point_x, 0
        )
        # quad emission for gen quantity under the split point
        gen_qty_quad_emission = np.minimum(
            gen_clear_qty, self.gen_emission_split_point_x
        )
        gen_emission_quad = (
            gen_qty_quad_emission * (self.gen_emission_coef_quad[:, 0] ** 2)
            + gen_qty_quad_emission * self.gen_emission_coef_quad[:, 1]
            + self.gen_emission_coef_quad[:, 2]
        )
        gen_emission_linear = (
            gen_qty_linear_emission * self.gen_emission_coef_linear[:, 0]
            + self.gen_emission_coef_linear[:, 1]
        )
        gen_emission = gen_emission_quad + gen_emission_linear
        # if gen clear quantity is 0, it does not launch, therefore it has no emission
        # fix the bug: constant term added on the emission
        gen_emission = [
            gen_emission[i] if gen_clear_qty[i] > 0.0 else 0.0
            for i in range(self.n_gens)
        ]
        return gen_emission


if __name__ == "__main__":
    engine = matlab.engine.start_matlab()
    elec_market = ElectricityMarket(
        Config,
        engine,
    )
    res = elec_market.run_step(1.0)
    elec_market.calc_carbon_emission(res)
