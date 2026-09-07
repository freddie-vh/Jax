import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax.numpy as jnp
from jax import grad, value_and_grad

def loss(params, x, target):
    a, b, c, d = params
    y = a * x**3 + b * x**2 + c * x + d
    return jnp.mean((y - target)**2)

def optimise():
    params = jnp.zeros(4)
    velocity = jnp.zeros(4)
    beta = 0.9
    x = jnp.linspace(-5, 5 ,100)
    target = 3 * x**3 + 4 * x**2 + 2 * x + 5
    lr = 0.0001
    loss_and_grad = value_and_grad(loss)
    for epoch in range(1000):
        loss_value, grads = loss_and_grad(params, x, target)
        velocity = beta * velocity + grads
        params -= velocity * lr
        if epoch % 100 == 0:
            print(loss_value)
    return params

print(optimise())
