import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

import numpy as np

X = mx.array([0, 0, 0, 1, 1, 0, 1, 1])
X = X.reshape(4, 2)
y = mx.array([0, 1, 1, 0])
y = y.reshape(4, 1)

class XOR(nn.Module):
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

    
xor = XOR()
mx.eval(xor.parameters())
vg = nn.value_and_grad(xor, loss_fn)
optim = opt.SGD(learning_rate=0.01)

for i in range(6000):
    j = mx.random.randint(0, 4)
    loss, grads = vg(xor, X[j], y[j])
    optim.update(xor, grads)
    mx.eval(xor.parameters(), optim.state)
    if not i % 500:
        print(f"Loss: {loss.item()}")

print("Trained model output: ")
print(np.round(xor(X)))
