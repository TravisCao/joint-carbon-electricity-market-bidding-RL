import numpy as np


class RuleAgent:
    def __init__(self):
        # Initialize any necessary variables or data structures
        pass

    def act(self, prev_emission, allowance, remaining_time):
        # output all gencon buying selling volumes
        # Implement the agent's decision-making logic based on the observation

        # if prev_emission has two dimension, it means there are multiple generators
        # return a multi-dimensional action
        if len(prev_emission.shape) > 1:
            prev_emission_total = np.sum(prev_emission, axis=0)
            prev_emission_mean = np.mean(prev_emission, axis=0)
        else:
            prev_emission_total = np.sum(prev_emission)
            prev_emission_mean = np.mean(prev_emission)

        remaining_allowance = allowance - prev_emission_total
        emission_expected = prev_emission_mean * remaining_time

        # if action > 0, buy extra allowance, else sell allowance
        action = np.array(emission_expected - remaining_allowance, dtype=np.float32)

        return action
