import jax.numpy as jnp
import jax.lax as lax
from jax import random
from jax import value_and_grad
import jax

@jax.jit(static_argnames=("steps", "unknown"))
def simulate(theta0, omega0, g, L, dt, steps, unknown):
    def forward_step(carry, _):
        theta, omega = carry
        angular_acceleration = -(g/L) * jnp.sin(theta)
        new_omega = omega + dt * angular_acceleration
        if unknown == True:
            new_omega -= 0.1 * omega
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
        difference = forward(params, new_state)
        theta_difference, omega_difference = difference
        new_state = jnp.array([new_theta + theta_difference, new_omega + omega_difference])
        return new_state, new_state
    final, result = lax.scan(forward_step, init=jnp.array([theta0, omega0]), xs=None, length=steps)
    initial = jnp.array([theta0, omega0])
    trajectories = jnp.concatenate([initial[None, :], result], axis=0)
    return trajectories

def init_params(layer_size, key):
    params = []
    for n_in, n_out in zip(layer_size[:-1], layer_size[1:]):
        key, w_key = random.split(key)
        w = random.normal(w_key, (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
        b = jnp.zeros(n_out)
        params.append((w,b))
    return params

def forward(params, x):
    *hidden, last = params
    for w, b in hidden:
        x = jax.nn.relu(x @ w + b)
    w, b = last
    return x @ w + b

def loss(params, x, target):
    y = forward(params, x)
    return jnp.mean((y - target)**2)

def optimise(x, target, params):
    s = jax.tree.map(jnp.zeros_like, params)
    velocity = s
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    loss_and_grad = value_and_grad(loss)
    for epoch in range(500):
        loss_value, grads = loss_and_grad(params, x, target)
        s = jax.tree.map(lambda s_i, g: beta2 * s_i + (1-beta2) * g**2, s, grads)
        velocity = jax.tree.map(lambda v_i, g: beta1 * v_i + (1-beta1) * g, velocity, grads)
        params = jax.tree.map(lambda p, v, s_i: p - lr * v / (jnp.sqrt(s_i) + 1e-8), params, velocity, s)
        if epoch % 100 == 0:
            print(loss_value)
    return params

def main():
    sim_data = simulate(0.5, 0.0, 9.81, 1.0, 0.01, 100, False) #generate simulation data without unknown component
    true_data = simulate(0.5, 0.0, 9.81, 1.0, 0.01, 100, True) #generate training data
    difference = true_data - sim_data
    key = random.PRNGKey(2)
    params = init_params((2,8,2), key)
    params = optimise(sim_data, difference, params)
    result = simulate_with_nn(0.5, 0.0, 9.81, 1.0, 0.01, 100, params)
    return result - true_data

print(main())

