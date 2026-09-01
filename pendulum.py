import jax.numpy as jnp
import jax.lax as lax
import jax

@jax.jit(static_argnames="steps")
def simulate(theta0, omega0, g, L, dt, steps):
    def forward_step(carry, x):
        theta, omega = carry
        angular_acceleration = -(g/L) * jnp.sin(theta)
        new_omega = omega + dt * angular_acceleration
        new_theta = theta + dt * new_omega
        new_state = jnp.array([new_theta, new_omega])
        return new_state, new_state
    final, result = lax.scan(forward_step, init=jnp.array([theta0, omega0]), xs=None, length=steps)
    initial = jnp.array([theta0, omega0])
    trajectories = jnp.concatenate([initial[None, :], result], axis=0)
    return trajectories

print(simulate(0.5, 0.0, 9.81, 1.0, 0.01, 10))