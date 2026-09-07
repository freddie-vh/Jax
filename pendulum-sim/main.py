import simulator
import network

from jax import random

def main():   
    sim_data = simulator.simulate(0.5, 0.0, 9.81, 1.0, 0.01, 100, False) #generate simulation data without unknown component
    true_data = simulator.simulate(0.5, 0.0, 9.81, 1.0, 0.01, 100, True) #generate training data
    difference = true_data - sim_data
    key = random.PRNGKey(2)
    params = network.init_params((2,16,2), key)
    params = network.optimise(sim_data, difference, params)
    result = simulator.simulate_with_nn(0.6, 0.0, 9.81, 1.5, 0.01, 100, params)
    return result - true_data

print(main())