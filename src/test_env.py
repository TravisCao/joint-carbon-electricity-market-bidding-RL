import numpy as np
from gencon import GenCon
from config import Config
from env import ElecMktEnv
import matlab.engine


def test_elec_mkt_env():
    # Create a Config object
    config = Config()
    engine = matlab.engine.start_matlab()

    # Create a GenCon object
    gen_con = GenCon(1, Config.gen_units[1], Config)

    # Create an ElecMktEnv object
    elec_mkt_env = ElecMktEnv(config, engine)

    # Test the reset method
    obs = elec_mkt_env.reset()
    assert isinstance(obs, np.ndarray)

    # Test the step method
    action = np.array([1, 2, 3, 4, 5])
    aciton = 1
    obs, r, terminated, info = elec_mkt_env.step(action)
    assert isinstance(obs, np.ndarray)
    assert isinstance(r, float)
    assert isinstance(terminated, bool)
    assert info is None

    # Test the get_state method
    state = elec_mkt_env.get_state()
    assert isinstance(state, np.ndarray)


def test_elec_mkt_env_gym():
    from ppo import make_env

    env = make_env("ElecMkt-v0", 0, False, "test", 0.99)
    next_obs, _ = env.reset()


test_elec_mkt_env_gym()
