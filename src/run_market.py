import matlab.engine
import numpy as np
import pandas as pd
import os
from tqdm import trange, tqdm

from config import Config
from gencon import GenCon
from markets import ElectricityMarket


# EM competitively, CM rule-based

agent_gen_action = 1.0
engine = matlab.engine.start_matlab()
non_renrewable_gens = [
    GenCon(i, Config.gen_units[i], Config) for i in range(Config.n_gens)
]
elec_market = ElectricityMarket(
    Config,
    engine,
)
emission_overall = np.zeros((48, 6))
for day in trange(1):
    emission = np.zeros((48, 6))
    for t in trange(Config.n_timesteps):
        res = elec_market.run_step(agent_gen_action)
        qtys = res[:, elec_market.QTY_COL]
        for i, gen in tqdm(enumerate((non_renrewable_gens))):
            e = gen.calc_carbon_emission(qtys[i])
            emission[t, i] = e
            print(
                f"day:{day} timestep: {elec_market.timestep} gen: {gen.gen_id}, emission: {e}"
            )
    emission_overall[day * 48 : (day + 1) * 48, :] = emission
np.savetxt(
    f"{os.path.dirname(os.path.abspath(__file__))}/../data/emission.csv",
    emission_overall,
    delimiter=",",
)


# if __name__ == "__main__":
#     for i in trange(100):
#         emission_overall[i : i + 48, :] = run_electricity_market_simple(1.0)
#     pd.DataFrame(emission_overall).to_csv("emission.csv")
