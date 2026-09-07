import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
from jax import grad, value_and_grad



# y = 2x + 3

def optimise():
    x = jnp.linspace(-5, 5, 100)
    target = 3*x**3 + 5*x**2 + 2*x + 3
    params = jnp.array([0.0, 0.0, 0.0, 0.0])
    lr1 = 0.0001
    lr2 = 0.001
    lr3 = 0.01
    lr4 = 0.1
    loss_and_grad = value_and_grad(loss)
    loss_value, grads = loss_and_grad(params, x, target)
    for epoch in range(1000):
        params -= jnp.array([
            lr1 * grads[0],
            lr2 * grads[1],
            lr3 * grads[2],
            lr4 * grads[3]  
        ])
        loss_value, grads = loss_and_grad(params, x, target)
        if epoch % 100 == 0:
            print(loss_value)
    return (params)

def loss(params, x, target):
    a, b, c, d = params
    y = a * x**3 + b * x**2 + c * x + d
    mse = jnp.mean((y - target)**2)
    return mse

print(optimise())
