import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax.numpy as jnp
import jax
from jax import grad, value_and_grad, random

def init_params(layer_size, key):
    params = []
    for n_in, n_out in zip(layer_size[:-1], layer_size[1:]):
        key, w_key = random.split(key)
        w = random.normal(w_key, (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
        b = jnp.zeros(n_out)
        params.append((w, b))
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

def optimise():
    key = random.PRNGKey(2)
    params = init_params([2,2,1], key)
    x = jnp.array([
        [0,0], 
        [0,1], 
        [1,0], 
        [1,1]
    ])
    target_1d = jnp.logical_xor(x[:,0], x[:,1]).astype(float)
    target = target_1d.reshape(-1, 1)
    s = jax.tree.map(jnp.zeros_like, params) #memory of squared gradients
    velocity = jax.tree.map(jnp.zeros_like, params)
    
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

params = optimise()
predictions = forward(params, jnp.array([[0,0], [0,1], [1,0], [1,1]]))
print(predictions)