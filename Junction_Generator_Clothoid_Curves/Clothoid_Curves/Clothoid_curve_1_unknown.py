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

phi = np.arctan2(Delta_y, Delta_x)
r = np.sqrt(Delta_x**2 + Delta_y**2)

Delta_Phi = v0 - phi

def G(A):
    def integrand(tau):
        return np.sin(A * tau**2 + (Delta_v - A) * tau + Delta_Phi)
    integral = quad(integrand, 0, 1)[0]
    return integral

def dG(A):
    def integrand(tau):
        return np.cos(A * tau**2 + (Delta_v - A) * tau + Delta_Phi) * (tau**2 - tau)
    integral = quad(integrand, 0, 1)[0]
    return integral

def newton(f, df, x0):
    iterates = [x0]
    for i in range(10):
        iterates.append(iterates[-1] - f(iterates[-1]) / df(iterates[-1]))
    return iterates

A = newton(G, dG, 1)[-1]

def H(A):
    def integrand(tau):
        return np.sin(A * tau**2 + (Delta_v - A) * tau + Delta_Phi + (np.pi/2))
    integral = quad(integrand, 0, 1)[0]
    return integral


L = r / H(A)
k0 = (Delta_v - A) / L
k1 = (2 * A) / L**2

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

x_vals, y_vals = clothoid_curve(x0, y0, v0, L, k0, k1)

plt.plot(x_vals, y_vals, label='Clothoid Curve')
plt.plot(x0, y0, 'ro', label='Start Point')
plt.plot(x1, y1, 'bo', label='End Point')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Clothoid Curve Connecting Two Points')
plt.grid(True)
plt.axis('equal')
plt.show()
