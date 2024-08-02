import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

import numpy as np

X = mx.array([1, 1, 0, 0, 1, 0, 0, 1]).reshape([4, 2])
y = mx.array([1, 1, 0, 0]).reshape([4, 1])

class XAND(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            nn.Linear(2, 2),
            nn.Linear(2, 1)
        ]
    
    def __call__(self, x):
        x = self.layers[0](x)
        x = mx.tanh(x)
        x = self.layers[1](x)
        return x
    
def loss_fn(model, input_data, expected_val):
    return mx.mean(mx.square(model(input_data) - expected_val))


xand = XAND()
mx.eval(xand.parameters())
vg = nn.value_and_grad(xand, loss_fn)
optim = opt.SGD(learning_rate=0.01)

for i in range(6000):
    j = mx.random.randint(0, 4)
    loss, grads = vg(xand, X[j], y[j])
    optim.update(xand, grads)
    mx.eval(xand.parameters(), optim.state)
    if not i % 500:
        print(f"Loss: {loss.item()}")

print("Trained model output: ")
print(np.round(xand(mx.array([[1, 1], [0, 0], [0, 0], [1, 0]]))))
