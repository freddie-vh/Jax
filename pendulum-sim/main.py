import simulator
import network

from jax import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def main():   
    sim_data = simulator.simulate(-0.5, 0.0, 9.81, 1.0, 0.01, 100, False) #generate simulation data without unknown component
    true_data = simulator.simulate(-0.5, 0.0, 9.81, 1.0, 0.01, 100, True) #generate training data
    difference = true_data - sim_data
    key = random.PRNGKey(2)
    params = network.init_params((2,16,2), key)
    params = network.optimise(sim_data, difference, params)
    result = simulator.simulate_with_nn(-0.5, 0.0, 9.81, 1.0, 0.01, 100, params)
    theta_true = true_data[:, 0]
    omega_true = true_data[:, 1]
    theta_sim = sim_data[:, 0]
    omega_sim = sim_data[:, 1]
    theta_network = result[:, 0]
    omega_network = result[:, 1]
    plt.scatter(theta_true, omega_true, label="True")
    plt.scatter(theta_sim, omega_sim, label="Sim")
    plt.scatter(theta_network, omega_network, label="Neural net")
    plt.xlabel("Angle")
    plt.ylabel("Angular velocity")
    plt.legend()
    plt.show()
    return result - true_data

print(main())