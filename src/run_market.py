from gencon import GenCon
from markets import ElectricityMarket
from config import Config
import matlab.engine


def run_electricity_market_simple(agent_gen_action):
    engine = matlab.engine.start_matlab()
    non_renrewable_gens = [
        GenCon(i, Config.gen_units[i], Config) for i in range(Config.n_gens)
    ]
    elec_market = ElectricityMarket(
        Config,
        engine,
    )
    for i in range(Config.n_timesteps):
        res = elec_market.run_step(agent_gen_action)
        qtys = res[:, elec_market.QTY_COL]
        for i, gen in enumerate(non_renrewable_gens):
            emission = gen.calc_carbon_emission(qtys[i])
            print(
                f"timestep: {elec_market.timestep} gen: {gen.gen_id}, emission: {emission}"
            )


if __name__ == "__main__":
    run_electricity_market_simple(1.2)
