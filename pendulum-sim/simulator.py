import network

import jax.numpy as jnp
import jax.lax as lax
from jax import random
import jax

@jax.jit(static_argnames=("steps", "unknown"))
def simulate(theta0, omega0, g, L, dt, steps, unknown):
    def forward_step(carry, _):
        theta, omega = carry
        angular_acceleration = -(g/L) * jnp.sin(theta)
        new_omega = omega + dt * angular_acceleration
        if unknown == True:
            new_omega -= 0.01 * omega
        new_theta = theta + dt * new_omega
        new_state = jnp.array([new_theta, new_omega])
        return new_state, new_state
    final, result = lax.scan(forward_step, init=jnp.array([theta0, omega0]), xs=None, length=steps)
    initial = jnp.array([theta0, omega0])
    trajectories = jnp.concatenate([initial[None, :], result], axis=0)
    return trajectories

@jax.jit(static_argnames="steps")
def simulate_with_nn(theta0, omega0, g, L, dt, steps, params):
    def forward_step(carry, _):
        theta, omega = carry
        angular_acceleration = -(g/L) * jnp.sin(theta)
        new_omega = omega + dt * angular_acceleration
        new_theta = theta + dt * new_omega
        new_state = jnp.array([new_theta, new_omega])
        difference = network.forward(params, new_state)
        return new_state, (new_state, difference)
    final, result = lax.scan(forward_step, init=jnp.array([theta0, omega0]), xs=None, length=steps)
    normal, difference = result
    trajectories = normal + difference
    initial = jnp.array([theta0, omega0])
    trajectories = jnp.concatenate([initial[None, :], trajectories], axis=0)
    return trajectories