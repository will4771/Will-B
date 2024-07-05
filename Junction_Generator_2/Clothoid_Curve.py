## 

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt

x0 = np.random.uniform(0, 10)
y0 = np.random.uniform(0, 10)
x1 = np.random.uniform(0, 10)
y1 = np.random.uniform(0, 10)

v0 = np.arctan2(y0, x0)
v1 = np.arctan2(y1, x1)

Delta_x = x1 - x0
Delta_y = y1 - y0
Delta_v = v1 - v0

def Ga(vars):
    L, A = vars

    def Integrand_x(tau):
        return np.cos(A * tau**2 + (Delta_v - A) * tau + v0)

    def Integrand_y(tau):
        return np.sin(A * tau**2 + (Delta_v - A) * tau + v0)

    integral_x = quad(Integrand_x, 0, 1)[0]
    integral_y = quad(Integrand_y, 0, 1)[0]

    Eq1 = Delta_x - L * integral_x
    Eq2 = Delta_y - L * integral_y

    return Eq1**2 + Eq2**2  # Squared sum to minimize L

# Initial guess for the variables (L, A)
initial_guess = [1, 1]

# Define bounds for L and A
bounds = [(None, None), (None, None)]  

# Perform minimization
result = minimize(Ga, initial_guess, bounds=bounds)
L, A = result.x

print(f"Optimized L: {L}, A: {A}")

# Function to find kappa0 and kappa1
def Kappa_finder(L, A):
    kappa0 = (Delta_v - A) / L
    kappa1 = 2 * A / L**2
    return kappa0, kappa1

k0, k1 = Kappa_finder(L, A)

# Function to generate clothoid curve
def clothoid_curve(x0, y0, theta0, L, kappa, kappa0, num_points=1000):
    s_vals = np.linspace(0, L, num_points)
    x_vals = np.zeros(num_points)
    y_vals = np.zeros(num_points)

    for i, s in enumerate(s_vals):
        def integrand_x(tau):
            return np.cos(0.5 * kappa0 * tau**2 + kappa * tau + theta0)

        def integrand_y(tau):
            return np.sin(0.5 * kappa0 * tau**2 + kappa * tau + theta0)

        x_vals[i] = x0 + quad(integrand_x, 0, s)[0]
        y_vals[i] = y0 + quad(integrand_y, 0, s)[0]

    return x_vals, y_vals

# Plotting
x_vals, y_vals = clothoid_curve(x0, y0, v0, L, k0, k1)

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label='Clothoid Curve')
plt.scatter([x0, x1], [y0, y1], color='red', zorder=5)
plt.text(x0, y0, 'Start', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
plt.text(x1, y1, 'End', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Clothoid Curve')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
