import jax.numpy as jnp
import jax.lax as lax
from jax import random
from jax import value_and_grad
import jax

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
    def training_step(carry, _):
        params, s, velocity = carry
        loss_value, grads = loss_and_grad(params, x, target)
        s = jax.tree.map(lambda s_i, g: beta2 * s_i + (1-beta2) * g**2, s, grads)
        velocity = jax.tree.map(lambda v_i, g: beta1 * v_i + (1-beta1) * g, velocity, grads)
        params = jax.tree.map(lambda p, v, s_i: p - lr * v / (jnp.sqrt(s_i) + 1e-8), params, velocity, s)
        return (params, s, velocity), loss_value
    init_carry = (params, s, velocity)
    (params, s, velocity), losses =  lax.scan(training_step, init_carry, xs=None, length=500)
    return params