import numpy as np
import matplotlib.pyplot as plt

# Endpoints
P0 = np.array([1, 1])
P1 = np.array([14, 5])

# Headings (in radians)  
theta0 = 0
theta1 = np.pi / 2

# Tangent vectors
T0 = np.array([np.cos(theta0), np.sin(theta0)])
T1 = np.array([np.cos(theta1), np.sin(theta1)])

# Distance between points
d = np.linalg.norm(P1 - P0)

# Tangent magnitudes
M0 = d * T0
M1 = d * T1

# Parametric cubic Hermite spline function
def hermite_curve(t, P0, P1, M0, M1):
    h00 = 2*t**3 - 3*t**2 + 1
    h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 = t**3 - t**2
    return h00*P0 + h10*M0 + h01*P1 + h11*M1

# Generate points on the curve
t_values = np.linspace(0, 1, 100)
curve_points = np.array([hermite_curve(t, P0, P1, M0, M1) for t in t_values])

# Plot the curve and the tangents
plt.plot(curve_points[:, 0], curve_points[:, 1], label="Cubic Hermite Spline")
plt.plot([P0[0], P0[0] + M0[0]], [P0[1], P0[1] + M0[1]], 'r--', label="Tangent at P0")
plt.plot([P1[0], P1[0] + M1[0]], [P1[1], P1[1] + M1[1]], 'g--', label="Tangent at P1")
plt.scatter([P0[0], P1[0]], [P0[1], P1[1]], color='black')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cubic Hermite Spline for Connection Road')
plt.grid(True)
plt.axis('equal')
plt.show()
