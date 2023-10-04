import matlab.engine
import numpy as np
import pandas as pd
import os
from tqdm import trange, tqdm

from config import Config
from gencon import GenCon
from agent import RuleAgent
from markets import ElectricityMarket, CarbonMarket


# EM competitively, CM rule-based

agent_gen_action = 1.0
engine = matlab.engine.start_matlab()
non_renewable_gens = [
    GenCon(i, Config.gen_units[i], Config) for i in range(Config.n_gens)
]
elec_market = ElectricityMarket(
    Config,
    engine,
)

days = 5

agent = RuleAgent()

carb_market = CarbonMarket(Config)
for day in range(days):
    emission = np.zeros((48, 6))
    # for t in trange(Config.n_timesteps):
    for t in range(Config.n_timesteps):
        res = elec_market.run_step(agent_gen_action)
        qtys = res[:, elec_market.QTY_COL]
        # for i, gen in tqdm(enumerate((non_renewable_gens))):
        #     e = gen.calc_carbon_emission(qtys[i])
        #     emission[t, i] = e
        # print(
        # f"day:{day} timestep: {elec_market.timestep} gen: {gen.gen_id}, emission: {e}"
        # )

    # carb_market.set_gen_emission(np.sum(emission, axis=0))
    # agent_action = agent.act(*carb_market.get_rule_obs())
    # print(agent_gen_action)
    # info = carb_market.trade(agent_action[0], agent_action[1])
    # print(info)


# np.savetxt(
#     f"{os.path.dirname(os.path.abspath(__file__))}/../data/emission.csv",
#     emission_overall,
#     delimiter=",",
# )


# if __name__ == "__main__":
#     for i in trange(100):
#         emission_overall[i : i + 48, :] = run_electricity_market_simple(1.0)
#     pd.DataFrame(emission_overall).to_csv("emission.csv")
