import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, quad
from scipy.optimize import minimize

def Para_plotter_blue(x, y):
    plt.plot(x, y, color='blue')

def Para_plotter_green(x, y):
    plt.plot(x, y, color='green')

def Inital_Incident_Road_Generator():
    x0 = np.random.uniform(25, 50)
    y0 = np.random.uniform(25, 50)
    Heading = np.random.uniform(-np.pi / 2, np.pi / 2)

    delta_x = np.cos(Heading)
    delta_y = np.sin(Heading)

    t = np.linspace(0, 5, 100)
    x = x0 - t * delta_x 
    y = y0 - t * delta_y

    return [x, y, Heading]

def clothoid_ode_rhs(state, s, kappa0, kappa1):
    x, y, theta = state[0], state[1], state[2]
    return [np.cos(theta), np.sin(theta), kappa0 + kappa1 * s]

def eval_clothoid(x0, y0, theta0, kappa0, kappa1, s):
    return odeint(clothoid_ode_rhs, [x0, y0, theta0], s, args=(kappa0, kappa1))

def Helper_Path_Gen(a, b, c, L):  # Length of roads
    x0, y0, theta0 = a, b, c
    kappa0, kappa1 = np.random.uniform(0.005, 0.025), np.random.uniform(0.005, 0.025)
    s = np.linspace(0, L, 1000)

    sol = eval_clothoid(x0, y0, theta0, kappa0, kappa1, s)

    xs, ys, thetas = sol[:, 0], sol[:, 1], sol[:, 2] 
    return xs, ys, thetas[-1] 

def Incident_Road_Gen(Helper_path):
    x0, y0, heading = Helper_path[0][-1], Helper_path[1][-1], Helper_path[2]
    delta_x = np.cos(heading)
    delta_y = np.sin(heading)

    t = np.linspace(0, 5, 100)
    x = x0 + t * delta_x 
    y = y0 + t * delta_y

    return [x, y, heading + np.pi]

def Incident_Map_Gen(Num):  # Change road length here
    Incident_Roads, Helper_Paths = [], []
    Incident_Roads.append(Inital_Incident_Road_Generator())
    
    i = 1
    while i <= Num:
        L = 10
        Helper_Paths.append(Helper_Path_Gen(Incident_Roads[i-1][0][0], Incident_Roads[i-1][1][0], Incident_Roads[i-1][2], L))
        Incident_Roads.append(Incident_Road_Gen(Helper_Paths[i-1]))
        
        should_pop = False
        for j in range(len(Incident_Roads) - 1):
            if np.abs(Incident_Roads[-1][0][0] - Incident_Roads[j][0][0]) <= 2.5 and np.abs(Incident_Roads[-1][1][0] - Incident_Roads[j][1][0]) <= 2.5:
                Incident_Roads.pop()
                Helper_Paths.pop()
                should_pop = True
                break
        
        if not should_pop:
            i += 1
        
    return Incident_Roads, Helper_Paths

def Incident_Map_Plot(Incident_Roads, Helper_paths):
    for i in range(len(Incident_Roads)):
        Para_plotter_blue(Incident_Roads[i][0], Incident_Roads[i][1])
    # Uncomment to plot helper paths if needed
    # for j in range(len(Helper_paths)):
    #     Para_plotter_green(Helper_paths[j][0], Helper_paths[j][1])

def Clothoid_Curve(a, b, c, d, heading0, heading1):
    x0, y0, x1, y1 = a, b, c, d
    v0, v1 = heading0, heading1 - np.pi
    Delta_x, Delta_y = x1 - x0, y1 - y0
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

    def clothoid_curve(x0, y0, theta0, L, kappa0, kappa1, num_points=1000):
        s_vals = np.linspace(0, L, num_points)
        x_vals = np.zeros(num_points)
        y_vals = np.zeros(num_points)
        for i, s in enumerate(s_vals):
            def integrand_x(tau):
                return np.cos(0.5 * kappa1 * tau**2 + kappa0 * tau + theta0)
            def integrand_y(tau):
                return np.sin(0.5 * kappa1 * tau**2 + kappa0 * tau + theta0)
            x_vals[i] = x0 + quad(integrand_x, 0, s)[0]
            y_vals[i] = y0 + quad(integrand_y, 0, s)[0]
        return [x_vals, y_vals]

    return clothoid_curve(x0, y0, v0, L, k0, k1)

def Connection_road_plotter(Connection_Roads):
    for road in Connection_Roads:
        plt.plot(road[0], road[1], color='red')

def Arg_min(Theta):
    while Theta > 2 * np.pi:
        Theta -= 2 * np.pi
    return Theta

def angle_fix(a):
    angle = Arg_min(a)
    return angle - np.pi if angle >= np.pi else angle + np.pi

def inital_Connection_road_gen(Incident_Roads, Helper_paths):
    Connection_Roads = []
    for i in range(len(Incident_Roads) - 1):
        Connection_Roads.append(Clothoid_Curve(Incident_Roads[0][0][0], Incident_Roads[0][1][0], Incident_Roads[i+1][0][0], Incident_Roads[i+1][1][0], Incident_Roads[0][2], angle_fix(Helper_paths[i][2])))

    return Connection_Roads

def Connection_road_gen(Incident_Roads, Helper_paths, Incident_N, Incident_Number): 
    Connection_Roads = []
    HEADING = Helper_paths[Incident_Number - 1][2]
    x = Incident_N[0][0]
    y = Incident_N[1][0]  # Corrected index to y-coordinate
    
    # First connection road
    Connection_Roads.append(Clothoid_Curve(
        x, 
        y, 
        Incident_Roads[0][0][0], 
        Incident_Roads[0][1][0], 
        angle_fix(HEADING), 
        Arg_min(Incident_Roads[0][2]) + 2 * np.pi
    ))

    # Remove the processed roads and helper paths
    del Incident_Roads[0]
    if Incident_Number > 1:  # Adjust index due to previous deletion
        del Incident_Roads[Incident_Number - 2]
    else:
        del Incident_Roads[0]  # If Incident_Number was 1
    del Helper_paths[Incident_Number - 1]

    # Generate connection roads for remaining incident roads
    for i in range(len(Incident_Roads)):
        Connection_Roads.append(Clothoid_Curve(
            x, 
            y, 
            Incident_Roads[i][0][0], 
            Incident_Roads[i][1][0], 
            Arg_min(HEADING) -np.pi, 
            Arg_min(Helper_paths[i][2]) -np.pi
        ))

    return Connection_Roads

# Generate and plot incident roads and connection roads
Incident_Roads, Helper_paths = Incident_Map_Gen(3)  # Number of incident roads +1
Incident_Map_Plot(Incident_Roads, Helper_paths)

# Initialize the list to store complete connection roads for each incident
Connection_Roads_Complete_Incident = []

# Generate initial connection roads
Incident_0_Connection_Roads = inital_Connection_road_gen(Incident_Roads, Helper_paths)
Connection_Roads_Complete_Incident.append(Incident_0_Connection_Roads)
Connection_road_plotter(Incident_0_Connection_Roads)

# Generate connection roads for remaining incidents
for i in range(1, len(Incident_Roads)):  # Start from 1 to avoid reprocessing the initial connection roads
    Incident_connection_Roads = Connection_road_gen(Incident_Roads.copy(), Helper_paths.copy(), Incident_Roads[i], i)
    Connection_Roads_Complete_Incident.append(Incident_connection_Roads)
    Connection_road_plotter(Incident_connection_Roads)

# Plot all roads
plt.show()

