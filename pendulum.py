import jax.numpy as jnp

class Simulator:
    def simulate(self, theta0, omega0, g, L, dt, steps):
        theta = [theta0]
        omega = [omega0]
        for i in range(0, steps):
            angular_acceleration = -(g/L) * jnp.sin(theta[i])
            omega.append(omega[i] + dt * angular_acceleration)
            theta.append(theta[i] + dt * omega[i+1])
            print(omega[i], theta[i])
            energy = 0.5 * L**2 * omega[i]**2 - g * L * jnp.cos(theta[i])
            print(energy)
        return list(zip(theta, omega))

mySimulator = Simulator()
mySimulator.simulate(0.5, 0.0, 9.81, 1.0, 0.01, 100)