# 110911542 Homework1
> this is the code
```
import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
y = np.array([1.9, 3.1, 3.9, 5.0, 6.2], dtype=np.float32)

def predict(a, xt):
    return a[0] + a[1] * xt

def MSE(a, x, y):
    total = 0
    for i in range(len(x)):
        total += (y[i] - predict(a, x[i])) ** 2
    return total

def loss(p):
    return MSE(p, x, y)

def optimize():
    p = [0.0, 0.0]  # Initial guess for p
    step_size = 0.01
    max_iterations = 1000

    for _ in range(max_iterations):
        current_loss = loss(p)

        # Perturb the parameters
        p_new = [p[0] + np.random.uniform(-step_size, step_size),
                 p[1] + np.random.uniform(-step_size, step_size)]
        new_loss = loss(p_new)

        # Update parameters if the new loss is smaller
        if new_loss < current_loss:
            p = p_new

    return p

p = optimize()

# Plot the graph
y_predicted = list(map(lambda t: p[0] + p[1] * t, x))
print('y_predicted=', y_predicted)
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, y_predicted, label='Fitted line')
plt.legend()
plt.show()
```
