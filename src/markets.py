from typing import Tuple, List

import matlab.engine
import numpy as np
import pandas as pd

from config import Config
from utils import get_logger


class ElectricityMarket:
    """Electricity market"""

    def __init__(
        self,
        config: Config,
        engine: matlab.engine,
    ) -> None:
        """
        Args:
            config (Config): configuration file
            engine (matlab.engine): matlab engine
        """
        self.engine = engine
        self.config = config
        self.LOAD_COEF = config.LOAD_COEF
        self.MAX_NEW_LOAD = config.MAX_NEW_LOAD
        self.gencost_coef = config.gencost_coef  # gen cost in generation
        self.n_gens = config.n_gens

        self.agent_gen_id = config.agent_gen_id

        self.WEIBUL_PAR = config.WEIBUL_PAR
        self.BETA_PAR1 = config.BETA_PAR1
        self.BETA_PAR2 = config.BETA_PAR2

        # load in each timestep
        self.loads = np.loadtxt(config.load_data_path) * self.LOAD_COEF
        # renews (wind and solar) in each timestep
        self.renews_exp = np.loadtxt(config.renew_data_path)
        # scaled loads, wind Pg exp., solar Pg exp.
        self.loads, self.wind_gen_exps, self.solar_gen_exps = self.process_load_gen()

        self.timestep = 0

        self.terminated = False

        # column index in run result
        self.PRC_COL = 1
        self.QTY_COL = 0

        self.LOG = get_logger()

    def reset_timestep(self):
        """reset timestep"""
        self.timestep = 0
        self.terminated = False

    def reset(self):
        self.reset_timestep()
        return np.array(
            [
                self.loads[self.timestep],
                self.wind_gen_exps[self.timestep],
                self.solar_gen_exps[self.timestep],
            ]
        )

    def get_state(self):
        return np.array(
            [
                self.loads[self.timestep],
                self.wind_gen_exps[self.timestep],
                self.solar_gen_exps[self.timestep],
                self.timestep,
            ]
        )

    def increase_timestep(self):
        """increase timestep and reset"""
        self.timestep += 1
        if self.timestep == len(self.loads):
            self.terminated = True

    def run_step(self, agent_gen_action: float):
        """run eletricity market in one timestep

        Args:
            agent_gen_action (float): gen action coef (bidding strategy)

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

        offers_qty, offers_prc, qty_prc_pairs = self.generate_piecewise_price(
            self.gencost_coef, agent_gen_action
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

    def calc_gen_reward(self, res):
        """calculate reward for a generator

        reward is price * quantity - cost

        Args:
            res (np.ndarray): market clearing result

        Returns:
            float: reward
        """
        return (
            res[self.agent_gen_id, self.QTY_COL] * res[self.agent_gen_id, self.PRC_COL]
            - res[self.agent_gen_id, -1]
        )

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

    def generate_piecewise_price(
        self, cof: np.ndarray, agent_gen_action: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """generate bidding strategy (price, vol) pair and gencost

        assumes only the agent gen take bidding action
        and other gens' bidding strategy are always 1

        Args:
            cof (np.ndarray): cof of gen cost
            agent_gen_action (float): agent_gen action (overall coefficient in gencost fn).
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
        tmp_cof[self.agent_gen_id, :] = cof[self.agent_gen_id, :] * agent_gen_action

        gmax = [self.config.gmax for _ in range(self.n_gens)]

        # price, vol pairs
        qty_prc_pairs = np.zeros((self.n_gens, n_piecewise * 2))
        for i in range(self.n_gens):
            cof1 = tmp_cof[i, 0]
            cof2 = tmp_cof[i, 1]
            cof3 = tmp_cof[i, 2]
            gen_gmax = gmax[i]
            for j in range(n_piecewise):
                cur_vol = gen_gmax / (n_piecewise - 1) * j
                cur_pri = cof1 * cur_vol**2 + cof2 * cur_vol + cof3
                qty_prc_pairs[i, 2 * j] = cur_vol
                qty_prc_pairs[i, 2 * j + 1] = cur_pri
        offers_qty = np.zeros((self.n_gens, n_piecewise - 1))
        offers_prc = np.zeros((self.n_gens, n_piecewise - 1))
        for i in range(n_piecewise - 1):
            offers_qty[:, i] = qty_prc_pairs[:, (i + 1) * 2] - qty_prc_pairs[:, i * 2]
            offers_prc[:, i] = (
                qty_prc_pairs[:, (i + 1) * 2 + 1] - qty_prc_pairs[:, i * 2 + 1]
            ) / offers_qty[:, i]
        return offers_qty, offers_prc, qty_prc_pairs

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


class CarbonMarket:
    """
    Carbon market
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.carbon_prices = [config.carbon_price_initial]
        self.carbon_allowance = [
            np.ones(config.n_gens) * config.carbon_allowance_initial
        ]
        self.gen_emissions = np.zeros((config.n_trading_days, config.n_gens))
        self.selling_volumes = np.zeros((config.n_trading_days, config.n_gens))
        self.buying_volumes = np.zeros((config.n_trading_days, config.n_gens))

        self.n_trading_days = config.n_trading_days
        self.day_t = 0

        self.price_alpha = config.price_alpha
        self.price_beta = config.price_beta

    @property
    def carbon_price_now(self) -> float:
        """current carbon price"""
        return self.carbon_prices[-1]

    @property
    def emission_price_ratio(self) -> float:
        """implement emission_price_ratio
        in paper
        "A hybrid interactive simulation method for studying emission trading behaviors"
        """
        recent_days_count = 7
        if self.day_t < recent_days_count:
            mean_prices_recent = np.mean(self.carbon_prices)
        else:
            mean_prices_recent = np.mean(self.carbon_prices[-recent_days_count:])
        return (self.carbon_price_now - mean_prices_recent) / mean_prices_recent

    @property
    def compliance_urgency_ratio(self) -> float:
        """implement compliance urgency ratio
        in paper
        "A hybrid interactive simulation method for studying emission trading behaviors"
        """
        T = self.n_trading_days
        t = self.day_t + 1

        # initial free emission allowance allocation
        q_e0 = self.config.carbon_allowance_initial * self.config.n_gens

        overall_emission_now = np.sum(self.gen_emissions)

        # TODO: check correctness
        trading_vols_now = np.sum(
            np.abs(self.selling_volumes - self.buying_volumes)[:t]
        )
        return ((T / (t - 1)) * overall_emission_now - trading_vols_now - q_e0) / (
            (T - (t - 1)) / (t - 1) * overall_emission_now
        ) + q_e0

    def price_clearing(self):
        """clear carbon price"""

        T = self.n_trading_days
        t = self.day_t + 1

        p_e_last = self.carbon_prices[-1]

        # assume we know the estimated total system emission

        # q_tilde_e_T is the estimated total system emission at time T

        if t == 1:
            q_tilde_e_T = T * np.sum(self.gen_emissions, axis=None)
        else:
            q_tilde_e_T = T / (t - 1) * np.sum(self.gen_emissions, axis=None)

        # overall supply of emission allowances set by regulator
        Q_e = self.config.carbon_allowance_initial * self.config.n_gens

        # long-term supply-demand balance
        long_term_balance = t / T * (q_tilde_e_T - Q_e) / Q_e

        # buying volume & selling volume last day

        q_b_e_last = np.sum(self.buying_volumes[t - 1, :])
        q_s_e_last = np.sum(self.selling_volumes[t - 1, :])

        # short_term supply-demand balance
        short_term_balance = (q_b_e_last - q_s_e_last) / (q_b_e_last + q_s_e_last)

        p_e_now = (
            p_e_last
            + self.price_alpha * long_term_balance
            + self.price_beta * short_term_balance
        )

        self.carbon_prices.append(p_e_now)

        return p_e_now

    def trade(
        self,
        buying_volumes: np.array,
        selling_volumes: np.array,
    ):
        """trade emission allowances

        Args:
            buying_volumes (np.array): buying volumes of each gen
            selling_volumes (np.array): selling volumes of each gen
        """

        assert sum(self.gen_emissions[self.day_t]) > 0

        # set volumes & gen emission
        self.buying_volumes[self.day_t, :] = buying_volumes
        self.selling_volumes[self.day_t, :] = selling_volumes

        # change price
        self.price_clearing()

        # change allowance
        carbon_allowance_last = self.carbon_allowance[-1]
        carbon_allowance_now = (
            np.array(carbon_allowance_last) + buying_volumes - selling_volumes
        )

        # trading days + 1
        self.day_t += 1

        self.carbon_allowance.append(carbon_allowance_now)

        info = {
            "allowance": carbon_allowance_now,
            "price": self.carbon_price_now,
            "remaining_days": self.n_trading_days - self.day_t,
        }

        return info

    def set_gen_emission(self, gen_emissions: np.array):
        """set gen emission"""
        self.gen_emissions[self.day_t] = gen_emissions

    def get_emission_record(self):
        """get previous emissions"""
        if self.day_t == 0:
            return self.gen_emissions[self.day_t]
        return self.gen_emissions[: self.day_t]

    def get_allowance(self):
        """get allowance"""
        return self.carbon_allowance[-1]

    def get_remaining_time(self):
        """get remaining time"""
        return self.n_trading_days - self.day_t
