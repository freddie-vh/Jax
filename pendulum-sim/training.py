import simulator
import network

from jax import random

def train(sim_params, network_params):
    key = random.PRNGKey(2)
    params = network.init_params(network_params, key)
    for training_cond in sim_params:
        sim_data = simulator.simulate(*training_cond, False)
        true_data = simulator.simulate(*training_cond, True)
        difference = true_data - sim_data
        params = network.optimise(sim_data, difference, params)
    return params