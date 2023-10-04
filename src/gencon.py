from typing import Tuple, Union

import numpy as np
import pandas as pd

from config import Config


class GenCon:
    """
    Generation company participating in electricity market
    and carbon market
    """

    def __init__(
        self,
        gen_id: int,
        gen_unit: str,
        config: Config,
    ) -> None:
        """

        Args:
            gen_id (int): gen id
            gen_unit (Union[str, None]): None if "non-renewable"
                                        else gen_unit str
            config (Config): config
        """
        self.gen_id = gen_id
        self.carbon_quota = 0.0
        self.carbon_emission = 0.0
        self.gen_unit = gen_unit
        self.config = config

        self.cems_df = pd.read_csv(config.cems_data_path)
        self.gen_emission_coef_quad = None
        self.gen_emission_coef_linear = None
        self.gen_emission_split_point_x = None
        self.gencost_coef = None
        self.setup_gen_coef()
        self.gmax = config.gmax

    def setup_gen_coef(self):
        """
        Set up gen coef based on gen_id and gen_type
        """
        gen_idx = self.cems_df[self.cems_df["unit"] == self.gen_unit].index[0]
        self.gen_emission_coef_quad = self.cems_df.iloc[gen_idx][
            ["coef-2-emission-quad", "coef-1-emission-quad", "coef-0-emission-quad"]
        ].values
        self.gen_emission_coef_linear = self.cems_df.iloc[gen_idx][
            ["coef-1-emission-linear", "coef-0-emission-linear"]
        ].values
        self.gen_emission_split_point_x = self.cems_df.iloc[gen_idx]["split-point-x"]

        self.gencost_coef = self.config.gencost_coef[self.gen_id]

    def calc_carbon_emission(self, gen_clear_qty: float) -> np.ndarray:
        """calculate carbon emission of gens based on generation gen_clear_qty
        the carbon emission is divided into quadratic and linear parts
        quadratic: gen emission under the split point x
        linear: gen emission over the split point x

        Args:
            gen_clear_qty (np.ndarray): generation gen_clear_qty in power system

        Returns:
            emissions(np.ndarray): carbon emission of each gen
        """

        # if gen clear quantity is 0, it does not launch, therefore it has no emission
        # fix the bug: constant term added on the emission
        if gen_clear_qty == 0.0:
            return 0.0
        # linear emission for gen quantity over the split point
        gen_qty_linear_emission = np.maximum(
            gen_clear_qty - self.gen_emission_split_point_x, 0
        )
        # quad emission for gen quantity under the split point
        gen_qty_quad_emission = np.minimum(
            gen_clear_qty, self.gen_emission_split_point_x
        )
        gen_emission_quad = (
            self.gen_emission_coef_quad[0] * (gen_qty_quad_emission**2)
            + self.gen_emission_coef_quad[1] * gen_qty_quad_emission
            + self.gen_emission_coef_quad[2]
        )
        gen_emission_linear = (
            self.gen_emission_coef_linear[0] * gen_qty_linear_emission
            + self.gen_emission_coef_linear[1]
        )
        gen_emission = gen_emission_quad + gen_emission_linear
        return gen_emission

    def generate_piecewise_price(
        self, gen_action: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """generate bidding strategy (price, vol) pair and gencost

        Args:
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
        tmp_cof = np.copy(self.gencost_coef)
        tmp_cof = self.gencost_coef * gen_action

        # price, vol pairs
        qty_prc_pairs = np.zeros(n_piecewise * 2)
        cof1, cof2, cof3 = (tmp_cof[i] for i in range(3))
        for j in range(n_piecewise):
            cur_vol = self.gmax / (n_piecewise - 1) * j
            cur_pri = cof1 * cur_vol**2 + cof2 * cur_vol + cof3
            qty_prc_pairs[2 * j] = cur_vol
            qty_prc_pairs[2 * j + 1] = cur_pri

        offers_qty = np.zeros(n_piecewise - 1)
        offers_prc = np.zeros(n_piecewise - 1)
        for i in range(n_piecewise - 1):
            offers_qty[i] = qty_prc_pairs[(i + 1) * 2] - qty_prc_pairs[i * 2]
            offers_prc[i] = (
                qty_prc_pairs[(i + 1) * 2 + 1] - qty_prc_pairs[i * 2 + 1]
            ) / offers_qty[i]
        return offers_qty, offers_prc, qty_prc_pairs


if __name__ == "__main__":
    gen = GenCon(1, Config.gen_units[2], Config)
    gen.calc_carbon_emission(1111)
    gen.generate_piecewise_price(1.5)
